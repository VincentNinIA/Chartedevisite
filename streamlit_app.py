import base64, io, json, re, requests
import streamlit as st
from notion_client import Client as Notion
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

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
MISTRAL_CHAT_MODEL= st.secrets.get("MISTRAL_CHAT_MODEL", "mistral-large-latest")

notion  = Notion(auth=NOTION_TOKEN)
mistral = MistralClient(api_key=MISTRAL_KEY)

# ===================
# HELPERS
# ===================
def _guess_data_url(image_bytes: bytes, fallback="image/jpeg") -> str:
    # Streamlit camera_input renvoie généralement JPEG.
    # Si besoin, détecte sommairement PNG.
    header = image_bytes[:8]
    mime = "image/png" if header.startswith(b"\x89PNG\r\n\x1a\n") else fallback
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"

def call_mistral_ocr(image_bytes: bytes) -> str:
    """
    OCR Mistral avec payload conforme:
    {
      "model": "...",
      "document": {"type":"image_url","image_url":"data:image/jpeg;base64,..."}
    }
    Retour: concat des pages[i].markdown
    """
    data_url = _guess_data_url(image_bytes)
    payload = {
        "model": MISTRAL_OCR_MODEL,
        "document": {"type": "image_url", "image_url": data_url},
        # "include_image_base64": False  # optionnel
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
    return "\n\n".join(p.get("markdown", "") for p in pages) or resp.get("text", "")

def naive_extract(text: str) -> dict:
    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone = re.search(r'(\+?\d[\d\s\.\-]{7,}\d)', text)
    website = re.search(r'(https?://[^\s]+|www\.[^\s]+)', text)

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
        "website": website.group(0) if website else None,
        "address": None
    }

def llm_structurize(rough: dict) -> dict:
    prompt = f"""
Tu reçois un dict Python avec des champs d'un lead issu d'une carte de visite.
Objectif: renvoyer STRICTEMENT un JSON avec clés:
full_name, company, job_title, email, phone (format E.164 si possible), website, address.
N'invente rien: mets null si inconnu. Corrige formats FR (tél).
Objet: {json.dumps(rough, ensure_ascii=False)}
Réponds UNIQUEMENT par le JSON.
"""
    resp = mistral.chat(
        model=MISTRAL_CHAT_MODEL,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        return rough

def generate_email_draft(lead: dict, notes: str) -> str:
    prompt = f"""
Tu es un SDR FR. Écris un email de prospection court (100-120 mots), personnalisé.
Lead: {json.dumps(lead, ensure_ascii=False)}
Contexte: {notes}
Objectif: proposer un échange de 15 minutes cette semaine (Teams ou téléphone).
Style: professionnel, concret, sans superlatifs ni pièce jointe.
"""
    resp = mistral.chat(
        model=MISTRAL_CHAT_MODEL,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()

def notion_find_page_by_email(email: str):
    """Recherche un lead existant par email dans la DB (anti-doublon simple)."""
    if not email:
        return None
    try:
        res = notion.databases.query(
            **{
                "database_id": NOTION_DB,
                "filter": {"property": "Email", "email": {"equals": email}}
            }
        )
        results = res.get("results", [])
        return results[0]["id"] if results else None
    except Exception:
        return None

def notion_upsert_lead(lead: dict, email_draft: str) -> str:
    props = {
        "Full Name": {"title":[{"text":{"content": lead.get("full_name") or "Lead (inconnu)"}}]},
        "Company": {"rich_text":[{"text":{"content": lead.get("company") or ""}}]},
        "Job Title": {"rich_text":[{"text":{"content": lead.get("job_title") or ""}}]},
        "Email": {"email": lead.get("email")},
        "Phone": {"phone": lead.get("phone")},
        "Website": {"url": lead.get("website")},
        "Address": {"rich_text":[{"text":{"content": lead.get("address") or ""}}]},
        "Source": {"select":{"name":"Business Card"}},
        "Status": {"select":{"name":"New"}},
    }
    existing_id = notion_find_page_by_email(lead.get("email"))
    if existing_id:
        page_id = existing_id
        notion.pages.update(page_id=page_id, properties=props)
    else:
        page = notion.pages.create(parent={"database_id": NOTION_DB}, properties=props)
        page_id = page["id"]

    if email_draft:
        notion.pages.update(page_id=page_id, properties={
            "Email draft": {"rich_text":[{"text":{"content": email_draft}}]}
        })
    return page_id

# ===================
# UI
# ===================
st.title("🪪 Carte → Notion (Mistral OCR + LLM)")
st.caption("Prends la carte en photo (mobile OK) ou dépose une image. Extraction + email brouillon automatiques.")

photo = st.camera_input("Prends la carte en photo")
notes = st.text_area("Notes (salon, besoins, contexte)", "")

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

    # 3) Brouillon d’email
    with st.spinner("Rédaction du brouillon d’email…"):
        draft = generate_email_draft(lead, notes)

    # 4) Notion (upsert)
    try:
        with st.spinner("Écriture dans Notion…"):
            page_id = notion_upsert_lead(lead, draft)
    except Exception as e:
        st.error(f"Notion en échec : {e}")
        st.stop()

    st.success("✅ Lead créé/mis à jour dans Notion + brouillon d’email ajouté.")
    st.subheader("Brouillon")
    st.code(draft, language="markdown")

    with st.expander("Texte OCR"):
        st.text(ocr_text)
else:
    st.caption("Autorise la caméra sur mobile, ou dépose un fichier.")
