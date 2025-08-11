# streamlit_app.py
import base64
import json
import re
import requests
import streamlit as st
from notion_client import Client as Notion
from mistralai import Mistral  # SDK récent

# ===================
# CONFIG
# ===================
st.set_page_config(page_title="Scan carte → Notion (Mistral)", page_icon="🪪", layout="centered")

NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
NOTION_DB    = st.secrets["NOTION_DB"]            # database_id
MISTRAL_KEY  = st.secrets["MISTRAL_API_KEY"]

# OCR (REST)
MISTRAL_OCR_URL   = st.secrets.get("MISTRAL_OCR_URL", "https://api.mistral.ai/v1/ocr")
MISTRAL_OCR_MODEL = st.secrets.get("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

# Chat (SDK)
MISTRAL_CHAT_MODEL = st.secrets.get("MISTRAL_CHAT_MODEL", "mistral-large-latest")
MISTRAL_AGENTS_URL  = st.secrets.get("MISTRAL_AGENTS_URL", "https://api.mistral.ai/v1")

# ----- NOMS EXACTS DE TES PROPRIÉTÉS NOTION -----
PROP_TITLE   = "Nom"                                   # title
PROP_STATUS  = "Statut"                                # status OU select
PROP_COMPANY = "Entreprise"                            # rich_text
PROP_EMAIL   = "E-mail"                                # email
PROP_PHONE   = "Téléphone"                             # phone_number
PROP_TOPIC   = "De quoi désirez-vous discuter ?"       # multi_select
PROP_NOTES   = "Parlez nous de vos besoins en quelques mots"            # rich_text (tes notes)

# Valeurs/Options attendues
TARGET_STATUS_VALUE = "Leads entrant"
TOPIC_OPTIONS = ["Formation", "Module", "Audit"]

# Notion client & Mistral
notion  = Notion(auth=NOTION_TOKEN)
mistral = Mistral(api_key=MISTRAL_KEY)

# ===================
# OCR
# ===================
def _guess_data_url(image_bytes: bytes, fallback="image/jpeg") -> str:
    header = image_bytes[:8]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif header[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    else:
        mime = fallback
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"

def call_mistral_ocr(image_bytes: bytes) -> str:
    data_url = _guess_data_url(image_bytes)
    payload = {
        "model": MISTRAL_OCR_MODEL,
        "document": {"type": "image_url", "image_url": data_url},
    }
    r = requests.post(
        MISTRAL_OCR_URL,
        headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    resp = r.json()
    pages = resp.get("pages", [])
    text = "\n\n".join(p.get("markdown", "") for p in pages).strip()
    return text or resp.get("text", "")

# ===================
# LLM
# ===================
def _chat_complete_text(prompt: str, temperature: float = 0.3) -> str:
    res = mistral.chat.complete(
        model=MISTRAL_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    # format récent
    try:
        return res.output[0].content[0].text.strip()
    except Exception:
        pass
    # fallback ancien
    try:
        return res.choices[0].message.content.strip()
    except Exception:
        return ""

# ===================
# Websearch Agent (Mistral Agents)
# ===================
def _create_websearch_agent() -> str:
    """
    Crée un agent Mistral avec le connecteur web_search et renvoie son agent_id.
    Doc: https://docs.mistral.ai/agents/connectors/websearch/
    """
    url = f"{MISTRAL_AGENTS_URL}/agents"
    payload = {
        "model": st.secrets.get("MISTRAL_AGENT_MODEL", MISTRAL_CHAT_MODEL),
        "name": "Websearch Agent",
        "description": "Agent capable de chercher sur le web des infos récentes (personne, entreprise).",
        "instructions": "Utilise le connecteur web_search pour chercher des informations factuelles et références.",
        "tools": [{"type": "web_search"}],
        "completion_args": {"temperature": 0.2, "top_p": 0.95},
    }
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {MISTRAL_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("id") or data.get("agent_id")


def get_websearch_agent_id() -> str | None:
    """Récupère/crée et met en cache l'agent_id en session."""
    key = "mistral_websearch_agent_id"
    if key in st.session_state and st.session_state[key]:
        return st.session_state[key]
    try:
        agent_id = _create_websearch_agent()
        st.session_state[key] = agent_id
        return agent_id
    except Exception:
        return None


def _ask_websearch_agent(agent_id: str, query: str) -> tuple[str, list[dict]]:
    """
    Démarre une conversation avec l’agent et renvoie (texte, références[]).
    Chaque référence: {title, url, source}
    """
    url = f"{MISTRAL_AGENTS_URL}/conversations"
    payload = {"agent_id": agent_id, "inputs": query, "stream": False}
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {MISTRAL_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    outputs = data.get("outputs", [])
    text_parts: list[str] = []
    refs: list[dict] = []
    for entry in outputs:
        if entry.get("type") == "message.output":
            for chunk in entry.get("content", []) or []:
                if chunk.get("type") == "text" and chunk.get("text"):
                    text_parts.append(chunk["text"]) 
                elif chunk.get("type") == "tool_reference" and chunk.get("url"):
                    refs.append({
                        "title": chunk.get("title") or chunk.get("url"),
                        "url": chunk.get("url"),
                        "source": chunk.get("source"),
                    })
    return ("\n".join(text_parts).strip() or "", refs)


def build_websearch_queries(lead: dict) -> list[str]:
    full_name = (lead.get("full_name") or "").strip()
    company = (lead.get("company") or "").strip()
    role = (lead.get("job_title") or "").strip()
    queries: list[str] = []
    if full_name and company:
        queries.append(f"Informations récentes et profil public de {full_name} ({role}) chez {company}.")
    elif full_name:
        queries.append(f"Profil public et informations professionnelles de {full_name} ({role}).")
    if company:
        queries.append(f"Résumé de {company} (activité, effectifs, actualités récentes, produits, stack, besoins).")
        queries.append(f"Opportunités d'usage de l'IA pour {company} par domaine (marketing, ops, support, data).")
    return [q for q in queries if q]


def run_web_enrichment(lead: dict, user_notes: str) -> tuple[str, list[dict]]:
    """
    Tente la recherche web via Mistral Agents; fallback: synthèse via LLM sans navigation.
    Renvoie (synthèse, références[]).
    """
    agent_id = get_websearch_agent_id()
    queries = build_websearch_queries(lead)
    if agent_id and queries:
        try:
            final_text: list[str] = []
            all_refs: list[dict] = []
            for q in queries:
                text, refs = _ask_websearch_agent(agent_id, q)
                if text:
                    final_text.append(text)
                if refs:
                    all_refs.extend(refs)
            if final_text:
                # Ajoute un cadrage pour l'IA
                final_text.append(
                    "\nSynthèse actionnable: propose des pistes concrètes où Nin-IA peut aider (formations, audits, modules) adaptées au contexte ci-dessus."
                )
                return ("\n\n".join(final_text), all_refs)
        except Exception:
            pass
    # Fallback: pas d'agent ou échec ⇒ synthèse sans navigation
    prompt = f"""
Tu vas générer un bref contexte exploitable sur ce lead puis des idées d'usage de l'IA.
Lead: {json.dumps(lead, ensure_ascii=False)}
Notes: {user_notes}
1) Contexte probable (sans inventer de faits précis non mentionnés) — hypothèses marquées explicitement.
2) 6 idées d'applications IA concrètes par domaine (marketing, ops, support, data, RH, produit), format liste.
"""
    return (_chat_complete_text(prompt, temperature=0.4), [])

def naive_extract(text: str) -> dict:
    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone = re.search(r'(\+?\d[\d\s\.\-]{7,}\d)', text)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    full_name = lines[0][:80] if lines else None
    job_title = next(
        (l[:80] for l in lines[:7] if re.search(
            r'(CEO|CTO|COO|CMO|Sales|Commercial|Marketing|Achat|Achats|RH|DRH|DG|Direction|Manager|Responsable|Chef de|Ingénieur|Consultant)',
            l, re.I)),
        None
    )
    company = next(
        (l[:80] for l in lines[:7]
         if l not in {full_name, job_title} and not re.search(r'@|https?://|www\.|\+?\d', l)),
        None
    )
    return {
        "full_name": full_name,
        "company": company,
        "job_title": job_title,
        "email": email.group(0) if email else None,
        "phone": (phone.group(0).replace(" ", "") if phone else None),
        "address": None
    }

def llm_structurize(rough: dict) -> dict:
    prompt = f"""
Tu reçois un dict Python avec des champs potentiellement incomplets d'un lead issu d'une carte de visite.
Objectif: renvoyer STRICTEMENT un JSON avec les clés:
full_name, company, job_title, email, phone (format E.164 si possible), address.
N'invente rien: mets null si inconnu. Corrige formats FR (tél).
Objet: {json.dumps(rough, ensure_ascii=False)}
Réponds UNIQUEMENT par le JSON.
"""
    content = _chat_complete_text(prompt, temperature=0.2)
    try:
        return json.loads(content)
    except Exception:
        return rough

def generate_email_draft(lead: dict, notes: str, web_context: str | None = None) -> str:
    prompt = f"""
Tu es mon assistant commercial, tu travailles chez Nin-IA on propose des formations IA, Audit IA et aussi des modules IA pour tout niveaux N8N à entrainement de RAG etc.
Tu recois les cartes de visite ainsi que des notes de clients potentiels que je rencontre.
Prends les données de ses notes pour personnalisés ton message.


Lead: {json.dumps(lead, ensure_ascii=False)}
Contexte: {notes}
Objectif: proposer une suite d'échange cette semaine pour lui présenter ce que l'on peut faire pour l'accompagner (Teams ou téléphone).
Style: professionnel, concret, avec une légère touche fun en fonction du ton de la note.
"""
    if web_context:
        prompt += f"\n\nContexte trouvé en ligne (synthèse):\n{web_context[:4000]}\n"
    return _chat_complete_text(prompt, temperature=0.6)

# ===================
# NOTION
# ===================
def notion_get_db_meta():
    return notion.databases.retrieve(NOTION_DB)

def notion_find_page_by_email(email: str):
    if not email:
        return None
    try:
        res = notion.databases.query(
            **{
                "database_id": NOTION_DB,
                "filter": {"property": PROP_EMAIL, "email": {"equals": email}}
            }
        )
        results = res.get("results", [])
        return results[0]["id"] if results else None
    except Exception:
        return None

def build_properties_for_upsert(lead: dict, topics: list, notes: str, db_meta: dict):
    props = {}

    # 1) Titre (Nom)
    props[PROP_TITLE] = {"title": [{"text": {"content": lead.get("full_name") or "Lead (inconnu)"}}]}

    # 2) Statut (status ou select) -> "Leads entrant"
    if PROP_STATUS in db_meta["properties"]:
        ptype = db_meta["properties"][PROP_STATUS]["type"]
        if ptype == "status":
            props[PROP_STATUS] = {"status": {"name": TARGET_STATUS_VALUE}}
        elif ptype == "select":
            props[PROP_STATUS] = {"select": {"name": TARGET_STATUS_VALUE}}

    # 3) Entreprise (rich_text)
    if lead.get("company") and PROP_COMPANY in db_meta["properties"] and db_meta["properties"][PROP_COMPANY]["type"] == "rich_text":
        props[PROP_COMPANY] = {"rich_text": [{"text": {"content": lead["company"]}}]}

    # 4) E-mail
    if lead.get("email") and PROP_EMAIL in db_meta["properties"] and db_meta["properties"][PROP_EMAIL]["type"] == "email":
        props[PROP_EMAIL] = {"email": lead["email"]}

    # 5) Téléphone
    if lead.get("phone") and PROP_PHONE in db_meta["properties"] and db_meta["properties"][PROP_PHONE]["type"] == "phone_number":
        props[PROP_PHONE] = {"phone_number": lead["phone"]}

    # 6) De quoi désirez-vous discuter ? (multi_select)
    if PROP_TOPIC in db_meta["properties"] and db_meta["properties"][PROP_TOPIC]["type"] == "multi_select":
        # On n'ajoute que les options autorisées
        valid = set(opt["name"] for opt in db_meta["properties"][PROP_TOPIC]["multi_select"]["options"])
        selected = [{"name": t} for t in topics if t in valid]
        props[PROP_TOPIC] = {"multi_select": selected}

    # 7) Notes (Parlez nous de vos besoins) en rich_text
    if notes and PROP_NOTES in db_meta["properties"] and db_meta["properties"][PROP_NOTES]["type"] == "rich_text":
        props[PROP_NOTES] = {"rich_text": [{"text": {"content": notes}}]}

    return props

def notion_upsert_lead(lead: dict, topics: list, notes: str) -> str:
    db_meta = notion_get_db_meta()
    props = build_properties_for_upsert(lead, topics, notes, db_meta)

    existing_id = notion_find_page_by_email(lead.get("email"))
    if existing_id:
        notion.pages.update(page_id=existing_id, properties=props)
        return existing_id
    else:
        page = notion.pages.create(parent={"database_id": NOTION_DB}, properties=props)
        return page["id"]

def notion_set_email_draft(page_id: str, draft: str):
    # Essaie de trouver une propriété rich_text probable pour le brouillon
    db_meta = notion_get_db_meta()
    candidates = [ "Email draft", "Brouillon", "Proposition d'email", "Email" ]
    lower_map = {k.lower(): k for k in db_meta["properties"].keys()}
    prop_name = None
    for c in candidates:
        k = lower_map.get(c.lower())
        if k and db_meta["properties"][k]["type"] == "rich_text":
            prop_name = k
            break
    if not prop_name:
        return  # pas de champ brouillon, on skip

    try:
        notion.pages.update(page_id=page_id, properties={
            prop_name: {"rich_text": [{"text": {"content": draft}}]}
        })
    except Exception:
        pass

# ===================
# UI
# ===================
st.title("🪪 Carte → Notion (Mistral OCR + LLM)")
st.caption("Prends la carte en photo (mobile OK) ou dépose une image. Statut = Leads entrant. Multi-sélection + Notes + Brouillon email.")

photo = st.camera_input("Prends la carte en photo")
notes = st.text_area("Parlez-nous de vos besoins", "")

topics = st.multiselect(
    "De quoi désirez-vous discuter ?",
    TOPIC_OPTIONS,
    default=[],
    help="Sera écrit dans la propriété multi-sélection."
)

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Ou téléverse une image", type=["png","jpg","jpeg"])
with col2:
    st.write("")  # placeholder

img = uploaded if uploaded is not None else photo

if st.button("Traiter") and img:
    img_bytes = img.getvalue()

    # 1) OCR
    try:
        with st.spinner("OCR Mistral…"):
            ocr_text = call_mistral_ocr(img_bytes)
            if not ocr_text.strip():
                raise RuntimeError("OCR vide.")
    except Exception as e:
        st.error(f"OCR en échec : {e}")
        st.stop()

    # 2) Extraction → normalisation
    rough = naive_extract(ocr_text)
    with st.spinner("Normalisation (LLM)…"):
        lead = llm_structurize(rough)

    # 3) Websearch (personne / entreprise) via Mistral Agents
    with st.spinner("Recherche web (Mistral)…"):
        web_summary, web_refs = run_web_enrichment(lead, notes)

    # 4) Notion (upsert + statut + sujets + notes)
    try:
        with st.spinner("Écriture dans Notion…"):
            page_id = notion_upsert_lead(lead, topics, notes)
    except Exception as e:
        st.error(f"Notion en échec : {e}")
        st.stop()

    # 5) Brouillon d’email (si champ approprié présent)
    with st.spinner("Rédaction du brouillon d’email…"):
        draft = generate_email_draft(lead, notes, web_context=web_summary or None)
        notion_set_email_draft(page_id, draft)

    st.success("✅ Lead créé/mis à jour (Statut = Leads entrant) + sujets + notes + brouillon ajouté.")
    st.subheader("Brouillon")
    st.code(draft, language="markdown")

    with st.expander("Texte OCR"):
        st.text(ocr_text)

    st.subheader("Contexte trouvé en ligne (Mistral Websearch)")
    if web_summary:
        st.markdown(web_summary)
    else:
        st.caption("Aucun résumé disponible.")
    if web_refs:
        st.caption("Sources")
        for ref in web_refs[:8]:
            title = ref.get("title") or ref.get("url")
            url = ref.get("url")
            src = ref.get("source")
            st.markdown(f"- [{title}]({url}){f' — {src}' if src else ''}")
else:
    st.caption("Autorise la caméra sur mobile, ou dépose un fichier.")
