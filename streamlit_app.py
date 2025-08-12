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

def llm_structurize(rough: Dict) -> Dict:
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
    # Titre
    props[PROP_TITLE] = {"title": [{"text": {"content": lead.get("full_name") or "Lead (inconnu)"}}]}
    # Statut
    if PROP_STATUS in db_meta["properties"]:
        ptype = db_meta["properties"][PROP_STATUS]["type"]
        if ptype == "status":
            props[PROP_STATUS] = {"status": {"name": TARGET_STATUS_VALUE}}
        elif ptype == "select":
            props[PROP_STATUS] = {"select": {"name": TARGET_STATUS_VALUE}}
    # Entreprise
    if lead.get("company") and PRO
