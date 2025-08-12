# streamlit_app.py
import base64
import json
import re
import requests
import streamlit as st
from typing import List, Dict, Optional, Tuple
from notion_client import Client as Notion
from mistralai import Mistral  # SDK récent
import unicodedata

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

# Agents (websearch)
MISTRAL_AGENTS_URL  = st.secrets.get("MISTRAL_AGENTS_URL", "https://api.mistral.ai/v1")
MISTRAL_AGENT_MODEL = st.secrets.get("MISTRAL_AGENT_MODEL", MISTRAL_CHAT_MODEL)

# ----- NOMS EXACTS DE TES PROPRIÉTÉS NOTION -----
PROP_TITLE        = "Nom"                                           # title
PROP_STATUS       = "Statut"                                        # status OU select
PROP_COMPANY      = "Entreprise"                                    # rich_text
PROP_EMAIL        = "E-mail"                                        # email
PROP_PHONE        = "Téléphone"                                     # phone_number
PROP_TOPIC        = "De quoi désirez-vous discuter ?"               # multi_select
PROP_NOTES        = "Parlez nous de vos besoins en quelques mots"   # rich_text
PROP_EMAIL_DRAFT  = "Brouillon à envoyer"                           # rich_text

# Valeurs/Options attendues
TARGET_STATUS_VALUE = "Leads entrant"
TOPIC_OPTIONS = ["Formation", "Module", "Audit IA"]

# Clients
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

def naive_extract(text: str) -> Dict:
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

# ---- Helpers nom complet → "NOM, Prénom"
def _guess_last_first(full_name: str):
    s = re.sub(r"\s+", " ", (full_name or "")).strip()
    if not s:
        return None, None
    # Forme "Nom, Prénom"
    if "," in s:
        left, right = [p.strip() for p in s.split(",", 1)]
        return left, right
    parts = s.split(" ")
    if len(parts) >= 2:
        # Si le NOM est en majuscules dans la carte
        uppers = [p for p in parts if p.isupper() and len(p) > 1]
        if uppers:
            last = " ".join(uppers)
            first = " ".join([p for p in parts if p not in uppers]) or None
            return last, first
        # Heuristique FR simple : dernier token = nom
        return parts[-1], " ".join(parts[:-1])
    return None, s

def format_lead_title(lead: Dict) -> str:
    last = (lead.get("last_name") or "").strip()
    first = (lead.get("first_name") or "").strip()
    company = (lead.get("company") or "").strip()
    if not (last and first) and lead.get("full_name"):
        glast, gfirst = _guess_last_first(lead["full_name"])
        last = last or (glast or "")
        first = first or (gfirst or "")
    name_part = f"{last.upper()}, {first}" if (last and first) else (lead.get("full_name") or "").strip()
    title = " - ".join([p for p in [name_part, company] if p]) or "Lead (inconnu)"
    return title[:200]

def llm_structurize(rough: Dict) -> Dict:
    prompt = f"""
Tu reçois un dict Python avec des champs potentiellement incomplets d'un lead issu d'une carte de visite.
Objectif: renvoyer STRICTEMENT un JSON avec les clés:
full_name, first_name, last_name, company, job_title, email, phone (format E.164 si possible), address.
N'invente rien: mets null si inconnu. Corrige formats FR (tél).
Objet: {json.dumps(rough, ensure_ascii=False)}
Réponds UNIQUEMENT par le JSON.
"""
    content = _chat_complete_text(prompt, temperature=0.2)
    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = dict(rough)  # fallback
    # Garantir les clés
    for k in ["full_name", "first_name", "last_name", "company", "job_title", "email", "phone", "address"]:
        parsed.setdefault(k, None)
    # Si first/last manquants → heuristique
    if not (parsed.get("first_name") and parsed.get("last_name")) and parsed.get("full_name"):
        glast, gfirst = _guess_last_first(parsed["full_name"])
        parsed["last_name"] = parsed.get("last_name") or glast
        parsed["first_name"] = parsed.get("first_name") or gfirst
    # Nettoyage téléphone simple
    if parsed.get("phone"):
        parsed["phone"] = re.sub(r"[^\d+]", "", parsed["phone"])
    return parsed

def generate_email_draft(lead: Dict, notes: str, web_context: Optional[str] = None) -> str:
    prompt = f"""
Tu es mon assistant commercial, tu travailles chez Nin-IA (formations IA, audits IA, modules IA).
Tu reçois des cartes de visite + notes de clients potentiels. Personnalise l'email en te basant surtout sur les notes.

Lead: {json.dumps(lead, ensure_ascii=False)}
Notes: {notes}
Objectif: proposer un échange (15 min) cette semaine (Teams ou téléphone).
Style: pro, concret, ton léger si les notes le permettent. 100-120 mots.
"""
    if web_context:
        prompt += f"\n\nContexte trouvé en ligne (synthèse, à utiliser seulement si pertinent):\n{web_context[:4000]}\n"
    return _chat_complete_text(prompt, temperature=0.6)

# ===================
# Agents (websearch)
# ===================
def _create_websearch_agent() -> Optional[str]:
    url = f"{MISTRAL_AGENTS_URL}/agents"
    payload = {
        "model": MISTRAL_AGENT_MODEL,
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

def get_websearch_agent_id() -> Optional[str]:
    key = "mistral_websearch_agent_id"
    if key in st.session_state and st.session_state[key]:
        return st.session_state[key]
    try:
        agent_id = _create_websearch_agent()
        st.session_state[key] = agent_id
        return agent_id
    except Exception:
        return None

def _ask_websearch_agent(agent_id: str, query: str) -> Tuple[str, List[Dict]]:
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
    text_parts: List[str] = []
    refs: List[Dict] = []
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

def build_websearch_queries(lead: Dict) -> List[str]:
    full_name = (lead.get("full_name") or "").strip()
    company = (lead.get("company") or "").strip()
    role = (lead.get("job_title") or "").strip()
    queries: List[str] = []
    if full_name and company:
        queries.append(f"Informations récentes et profil public de {full_name} ({role}) chez {company}.")
    elif full_name:
        queries.append(f"Profil public et informations professionnelles de {full_name} ({role}).")
    if company:
        queries.append(f"Résumé de {company} (activité, effectifs, actualités récentes, produits).")
        queries.append(f"Opportunités d'usage de l'IA pour {company} (marketing, ops, support, data).")
    return [q for q in queries if q]

def run_web_enrichment(lead: Dict, user_notes: str) -> Tuple[str, List[Dict]]:
    agent_id = get_websearch_agent_id()
    queries = build_websearch_queries(lead)
    if agent_id and queries:
        try:
            final_text: List[str] = []
            all_refs: List[Dict] = []
            for q in queries:
                text, refs = _ask_websearch_agent(agent_id, q)
                if text:
                    final_text.append(text)
                if refs:
                    all_refs.extend(refs)
            if final_text:
                final_text.append(
                    "\nSynthèse actionnable: propose des pistes concrètes où Nin-IA peut aider (formations, audits, modules)."
                )
                return ("\n\n".join(final_text), all_refs)
        except Exception:
            pass
    # Fallback sans web
    prompt = f"""
Tu vas générer un bref contexte exploitable sur ce lead puis des idées d'usage de l'IA.
Lead: {json.dumps(lead, ensure_ascii=False)}
Notes: {user_notes}
1) Contexte probable (hypothèses explicites si nécessaire).
2) 6 idées d'applications IA concrètes par domaine (marketing, ops, support, data, RH, produit), liste.
"""
    return (_chat_complete_text(prompt, temperature=0.4), [])

# ===================
# NOTION – utilitaires schéma
# ===================
def notion_get_db_meta() -> Dict:
    return notion.databases.retrieve(NOTION_DB)

def _normalize_name(name: str) -> str:
    if not name:
        return ""
    nf = unicodedata.normalize("NFD", name)
    without_accents = "".join(ch for ch in nf if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", without_accents).strip().lower()

def _find_property_key(target: str, properties: Dict) -> Optional[str]:
    if target in properties:
        return target
    lower_map = {k.lower(): k for k in properties.keys()}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    norm_target = _normalize_name(target)
    norm_map = {_normalize_name(k): k for k in properties.keys()}
    return norm_map.get(norm_target)

def ensure_db_schema():
    """Garantit que les propriétés clés existent avec le bon type; ajoute options si manquantes."""
    meta = notion_get_db_meta()
    props = meta.get("properties", {})

    def ensure_rich_text(name: str):
        if name in props and props[name]["type"] == "rich_text":
            return
        notion.databases.update(database_id=NOTION_DB, properties={name: {"rich_text": {}}})

    def ensure_email(name: str):
        if name in props and props[name]["type"] == "email":
            return
        notion.databases.update(database_id=NOTION_DB, properties={name: {"email": {}}})

    def ensure_phone(name: str):
        if name in props and props[name]["type"] == "phone_number":
            return
        notion.databases.update(database_id=NOTION_DB, properties={name: {"phone_number": {}}})

    def ensure_multi_select(name: str, options: List[str]):
        nonlocal props
        if name not in props or props[name]["type"] != "multi_select":
            notion.databases.update(
                database_id=NOTION_DB,
                properties={name: {"multi_select": {"options": [{"name": o} for o in options]}}}
            )
        else:
            current = set(opt["name"] for opt in props[name]["multi_select"]["options"])
            missing = [o for o in options if o not in current]
            if missing:
                new_opts = props[name]["multi_select"]["options"] + [{"name": o} for o in missing]
                notion.databases.update(
                    database_id=NOTION_DB,
                    properties={name: {"multi_select": {"options": new_opts}}}
                )

    def ensure_status(name: str, required_value: str):
        if name not in props:
            notion.databases.update(
                database_id=NOTION_DB,
                properties={name: {"select": {"options": [{"name": required_value}]}}}
            )
            return
        p = props[name]
        if p["type"] == "status":
            cur = set(opt["name"] for opt in p["status"]["options"])
            if required_value not in cur:
                new_opts = p["status"]["options"] + [{"name": required_value}]
                notion.databases.update(
                    database_id=NOTION_DB,
                    properties={name: {"status": {"options": new_opts}}}
                )
        elif p["type"] == "select":
            cur = set(opt["name"] for opt in p["select"]["options"])
            if required_value not in cur:
                new_opts = p["select"]["options"] + [{"name": required_value}]
                notion.databases.update(
                    database_id=NOTION_DB,
                    properties={name: {"select": {"options": new_opts}}}
                )
        else:
            notion.databases.update(
                database_id=NOTION_DB,
                properties={name: {"select": {"options": [{"name": required_value}]}}}
            )

    ensure_status(PROP_STATUS, TARGET_STATUS_VALUE)
    ensure_email(PROP_EMAIL)
    ensure_phone(PROP_PHONE)
    ensure_rich_text(PROP_COMPANY)
    ensure_rich_text(PROP_NOTES)
    ensure_rich_text(PROP_EMAIL_DRAFT)
    ensure_multi_select(PROP_TOPIC, TOPIC_OPTIONS)

# Exécuter l’assurance schéma au démarrage
try:
    ensure_db_schema()
except Exception as e:
    st.warning(f"Impossible de garantir le schéma Notion (droits/partage ?) : {e}")

# ===================
# NOTION – CRUD
# ===================
def notion_find_page_by_email(email: Optional[str]) -> Optional[str]:
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

def build_properties_for_upsert(lead: Dict, topics: List[str], notes: str, db_meta: Dict) -> Dict:
    props: Dict = {}

    # ---- Titre : "NOM, Prénom - Entreprise"
    title_text = format_lead_title(lead)
    props[PROP_TITLE] = {"title": [{"text": {"content": title_text}}]}

    # Statut
    if PROP_STATUS in db_meta["properties"]:
        ptype = db_meta["properties"][PROP_STATUS]["type"]
        if ptype == "status":
            props[PROP_STATUS] = {"status": {"name": TARGET_STATUS_VALUE}}
        elif ptype == "select":
            props[PROP_STATUS] = {"select": {"name": TARGET_STATUS_VALUE}}

    # Entreprise
    if lead.get("company") and PROP_COMPANY in db_meta["properties"] and db_meta["properties"][PROP_COMPANY]["type"] == "rich_text":
        props[PROP_COMPANY] = {"rich_text": [{"text": {"content": lead["company"]}}]}

    # E-mail
    if lead.get("email") and PROP_EMAIL in db_meta["properties"] and db_meta["properties"][PROP_EMAIL]["type"] == "email":
        props[PROP_EMAIL] = {"email": lead["email"]}

    # Téléphone
    if lead.get("phone") and PROP_PHONE in db_meta["properties"] and db_meta["properties"][PROP_PHONE]["type"] == "phone_number":
        props[PROP_PHONE] = {"phone_number": lead["phone"]}

    # Sujet (multi-select)
    if PROP_TOPIC in db_meta["properties"] and db_meta["properties"][PROP_TOPIC]["type"] == "multi_select":
        valid = set(opt["name"] for opt in db_meta["properties"][PROP_TOPIC]["multi_select"]["options"])
        selected = [{"name": t} for t in topics if t in valid]
        props[PROP_TOPIC] = {"multi_select": selected}

    # Notes
    if notes and PROP_NOTES in db_meta["properties"] and db_meta["properties"][PROP_NOTES]["type"] == "rich_text":
        props[PROP_NOTES] = {"rich_text": [{"text": {"content": notes}}]}

    return props

def notion_upsert_lead(lead: Dict, topics: List[str], notes: str) -> str:
    db_meta = notion_get_db_meta()
    props = build_properties_for_upsert(lead, topics, notes, db_meta)
    existing_id = notion_find_page_by_email(lead.get("email"))
    if existing_id:
        notion.pages.update(page_id=existing_id, properties=props)
        return existing_id
    page = notion.pages.create(parent={"database_id": NOTION_DB}, properties=props)
    return page["id"]

def _to_rich_text_chunks(text: str, chunk_size: int = 1800) -> List[Dict]:
    chunks: List[Dict] = []
    if not text:
        return [{"text": {"content": ""}}]
    for i in range(0, len(text), chunk_size):
        segment = text[i : i + chunk_size]
        chunks.append({"text": {"content": segment}})
    return chunks

def notion_set_email_draft(page_id: str, draft: str) -> Tuple[bool, Optional[str]]:
    try:
        page = notion.pages.update(
            page_id=page_id,
            properties={PROP_EMAIL_DRAFT: {"rich_text": _to_rich_text_chunks(draft)}},
        )
        return True, None
    except Exception as e:
        return False, str(e)

def notion_append_email_draft_block(page_id: str, draft: str) -> Tuple[bool, Optional[str]]:
    try:
        heading_block = {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Brouillon à envoyer"}}]},
        }
        code_block = {
            "object": "block",
            "type": "code",
            "code": {
                "language": "markdown",
                "rich_text": [{"type": "text", "text": {"content": chunk["text"]["content"]}} for chunk in _to_rich_text_chunks(draft)],
            },
        }
        notion.blocks.children.append(block_id=page_id, children=[heading_block, code_block])
        return True, None
    except Exception as e:
        return False, str(e)

# ===================
# UI
# ===================
st.title("🪪 Carte → Notion (Mistral OCR + LLM)")
st.caption("Photo (mobile OK) → OCR → Lead Notion (Statut = Leads entrant) → Sujets → Notes → Brouillon email.")

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
    st.write("")

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

    # 3) Websearch (optionnel)
    with st.spinner("Recherche web (Mistral)…"):
        web_summary, web_refs = run_web_enrichment(lead, notes)

    # 4) Notion (upsert + statut + sujets + notes)
    try:
        with st.spinner("Écriture dans Notion…"):
            page_id = notion_upsert_lead(lead, topics, notes)
    except Exception as e:
        st.error(f"Notion en échec : {e}")
        st.stop()

    # 5) Brouillon d’email
    with st.spinner("Rédaction du brouillon d’email…"):
        draft = generate_email_draft(lead, notes, web_context=web_summary or None)
        ok, err = notion_set_email_draft(page_id, draft)
        if not ok:
            st.warning(f"Propriété '{PROP_EMAIL_DRAFT}' indisponible ({err}). Ajout dans le corps de la page…")
            ok2, err2 = notion_append_email_draft_block(page_id, draft)
            if ok2:
                st.info("Brouillon ajouté dans le corps de la page (bloc).")
            else:
                st.error(f"Impossible d'ajouter le brouillon : {err2}")

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
