import streamlit as st
import base64, io, re, json, requests
from notion_client import Client as Notion

# --- CONFIG ---
st.set_page_config(page_title="Scan carte → Notion", page_icon="🪪", layout="centered")

NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
NOTION_DB    = st.secrets["NOTION_DB"]     # id de la DB leads
MISTRAL_KEY  = st.secrets["MISTRAL_API_KEY"]
# Optionnel : un LLM texte pour l’extraction JSON + l’email (peut être Mistral aussi)
LLM_API_KEY  = st.secrets.get("LLM_API_KEY", MISTRAL_KEY)

notion = Notion(auth=NOTION_TOKEN)

# --- HELPERS ---

def call_mistral_ocr(image_bytes: bytes, lang: str = "fr") -> str:
    """
    Appelle ton OCR Mistral et renvoie du texte brut/markdown.
    Adapte l'endpoint et le payload à ta version OCR.
    """
    b64 = base64.b64encode(image_bytes).decode()
    url = st.secrets.get("MISTRAL_OCR_URL", "https://api.mistral.ai/v1/ocr")  # <-- ajuste

    headers = {
        "Authorization": f"Bearer {MISTRAL_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": st.secrets.get("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
        "input": {"image_base64": b64, "lang": lang}
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Adapte selon le schéma renvoyé par l’OCR Mistral
    # Exemple attendu: {"text": "..."} ou {"output":{"text":"..."}}
    return data.get("text") or data.get("output", {}).get("text", "")

def naive_extract_fields(text: str) -> dict:
    """Extraction minimale sans LLM (fallback). Tu peux remplacer par un appel LLM."""
    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    phone = re.search(r'(\+?\d[\d\s\.\-]{7,}\d)', text)
    website = re.search(r'(https?://[^\s]+|www\.[^\s]+)', text)

    # Heuristiques basiques nom/société/poste (à raffiner)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    full_name = lines[0][:80] if lines else None
    company   = None
    job_title = None
    # Cherche une ligne avec mots clés de poste (fr/en)
    for l in lines[:6]:
        if re.search(r'(CEO|CTO|COO|CMO|Sales|Commercial|Marketing|Achat|Achats|RH|DRH|DG|Direction|Manager|Responsable|Chef de|Ingénieur|Consultant)', l, re.I):
            job_title = l[:80]
            break
    # Société : prend une ligne haute différente du nom/poste/email
    for l in lines[:6]:
        if l != full_name and l != job_title and not re.search(r'@|https?://|www\.|\+?\d', l):
            company = l[:80]
            break

    return {
        "full_name": full_name,
        "company": company,
        "job_title": job_title,
        "email": email.group(0) if email else None,
        "phone": phone.group(0).replace(" ", "") if phone else None,
        "website": website.group(0) if website else None,
        "address": None
    }

def create_or_update_lead_notion(lead: dict, raw_confidence: float = None, email_draft: str = "") -> str:
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
    if raw_confidence is not None:
        props["OCR Confidence"] = {"number": float(raw_confidence)}

    page = notion.pages.create(parent={"database_id": NOTION_DB}, properties=props)
    if email_draft:
        notion.pages.update(page_id=page["id"], properties={
            "Email draft": {"rich_text":[{"text":{"content": email_draft}}]}
        })
    return page["id"]

def generate_email_draft(lead: dict, notes: str) -> str:
    """
    Remplace par ton appel LLM (Mistral chat/ OpenAI/Anthropic).
    Ici : prompt simple côté serveur via Mistral chat completions.
    """
    url = st.secrets.get("LLM_CHAT_URL", "https://api.mistral.ai/v1/chat/completions")
    model = st.secrets.get("LLM_MODEL", "mistral-large-latest")
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = f"""
Tu es un SDR FR. Rédige un email de 100-120 mots, clair et personnalisé.
Lead: {json.dumps(lead, ensure_ascii=False)}
Contexte: {notes}
Objectif: proposer un échange de 15 minutes (Teams ou téléphone) cette semaine.
Style: professionnel, sans superlatifs, pas de pièces jointes, pas de jargon.
"""
    payload = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
        "temperature": 0.6
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Adapte au schéma de ta réponse (message.content)
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Bonjour,\n\nRavi de notre échange. Nous aidons des PME à automatiser la gestion des leads et la saisie via IA. Partant pour 15 minutes cette semaine pour valider vos besoins ?\n\nBien à vous,\n—"

# --- UI ---

st.title("🪪 Carte → Lead Notion → Brouillon d’email")

photo = st.camera_input("Prends la carte en photo")  # marche aussi sur mobile
notes = st.text_area("Notes (salon, besoins, contexte)", "")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Ou téléverse une image", type=["png","jpg","jpeg"])
with col2:
    lang = st.selectbox("Langue OCR", ["fr","en"], index=0)

img = uploaded if uploaded is not None else photo

if st.button("Traiter") and img:
    img_bytes = img.getvalue()
    # 1) OCR
    try:
        with st.spinner("OCR (Mistral)…"):
            text = call_mistral_ocr(img_bytes, lang=lang)
    except Exception as e:
        st.error(f"OCR en échec : {e}")
        st.stop()

    # 2) Extraction (fallback naïf; remplace par LLM si tu veux)
    lead = naive_extract_fields(text)

    # 3) Email draft
    with st.spinner("Rédaction du brouillon d’email…"):
        email_draft = generate_email_draft(lead, notes)

    # 4) Notion
    try:
        with st.spinner("Création dans Notion…"):
            page_id = create_or_update_lead_notion(lead, raw_confidence=None, email_draft=email_draft)
    except Exception as e:
        st.error(f"Notion en échec : {e}")
        st.stop()

    st.success("✅ Lead créé dans Notion + brouillon d’email ajouté.")
    st.subheader("Brouillon")
    st.code(email_draft, language="markdown")

    with st.expander("Texte OCR"):
        st.text(text)

else:
    st.caption("Autorise la caméra sur mobile, ou dépose un fichier.")
