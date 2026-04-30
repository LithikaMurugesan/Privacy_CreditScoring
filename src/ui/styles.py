import streamlit as st


def apply_custom_styles():
    """
    Apply custom CSS styles to the Streamlit application.
    
    This function injects custom CSS into the Streamlit app to provide:
      - Professional typography (IBM Plex Sans/Mono)
      - Dark theme with warm color palette
      - Custom login page styling
      - Sidebar navigation styling
      - Reusable component styles (KPI cards, log boxes, mode cards)
    
    Call this function once at the start of your Streamlit app,
    after st.set_page_config().
    
    Returns:
        None
    """
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
        unsafe_allow_html=True,
    )
    
   
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

/* Original styling */
html, body, [class*="css"] { 
    font-family: 'IBM Plex Sans', sans-serif; 
}
code, pre { 
    font-family: 'IBM Plex Mono', monospace !important; 
}

/* Original topbar */
.topbar { 
    background: linear-gradient(90deg, #0f172a, #1e3a5f); 
    color: white; 
    padding: 18px 28px; 
    border-radius: 12px; 
    margin-bottom: 20px; 
}
.topbar h1 { 
    margin: 0; 
    font-size: 1.4rem; 
    font-weight: 700; 
}
.topbar p { 
    margin: 0; 
    font-size: 0.82rem; 
    color: #94a3b8; 
}

/* Original KPI cards */
.kpi { 
    background: #0f172a; 
    border: 1px solid #1e3a5f; 
    border-radius: 10px; 
    padding: 16px; 
    text-align: center; 
}
.kpi-val { 
    font-size: 1.8rem; 
    font-weight: 700; 
    color: #38bdf8; 
}
.kpi-lbl { 
    font-size: 0.78rem; 
    color: #64748b; 
    margin-top: 4px; 
}

/* Original log box */
.log-box { 
    background: #0f172a; 
    border: 1px solid #1e3a5f; 
    border-radius: 8px; 
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace; 
    font-size: 0.78rem; 
    color: #94a3b8;
    max-height: 320px; 
    overflow-y: auto; 
}

/* Original mode card */
.mode-card { 
    background: #0f172a; 
    border: 1px solid #334155; 
    border-radius: 10px; 
    padding: 14px 18px; 
    margin: 6px 0; 
}
.mode-card h4 { 
    margin: 0 0 4px; 
    color: #38bdf8; 
    font-size: 0.95rem; 
}
.mode-card p { 
    margin: 0; 
    font-size: 0.82rem; 
    color: #94a3b8; 
}

/* Sidebar - fixed collapse behavior */
section[data-testid="stSidebar"] {
    min-width: 80px !important;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 280px !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 80px !important;
}

/* Sidebar buttons - original colors */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: #94a3b8;
    border: 1px solid #1e293b;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.25s ease;
    padding: 9px 14px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border-color: #38bdf8;
}

/* ===== LOGIN PAGE ===== */

/* Hide sidebar and header when login-page marker is present */
.stApp:has(.login-page-marker) header[data-testid="stHeader"],
.stApp:has(.login-page-marker) section[data-testid="stSidebar"],
.stApp:has(.login-page-marker) .stDeployButton {
    display: none !important;
}

/* Light gradient background */
.stApp:has(.login-page-marker) {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #f8fafc 100%) !important;
}

/* Center everything */
.stApp:has(.login-page-marker) .main,
.stApp:has(.login-page-marker) [data-testid="stAppViewContainer"],
.stApp:has(.login-page-marker) section[data-testid="stMain"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 100vh !important;
}

.stApp:has(.login-page-marker) .block-container,
.stApp:has(.login-page-marker) [data-testid="block-container"] {
    padding: 2rem !important;
    max-width: 450px !important;
    width: 100% !important;
}

/* Clean white card */
.stApp:has(.login-page-marker) [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 2.5rem 2rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    animation: cardSlideUp 0.5s ease-out;
    width: 100% !important;
}

@keyframes cardSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Seamless form */
.stApp:has(.login-page-marker) [data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* Brand section */
.lc-brand { text-align: center; margin-bottom: 1.5rem; }
.lc-icon {
    width: 56px; height: 56px;
    margin: 0 auto 1rem;
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
}
.lc-icon i { font-size: 1.6rem; color: #ffffff; }
.lc-brand h2 {
    color: #1e293b; font-size: 1.5rem; font-weight: 700;
    margin: 0 0 0.3rem; letter-spacing: -0.02em;
}
.lc-brand p { color: #64748b; font-size: 0.9rem; margin: 0; }

/* Input styling */
.stApp:has(.login-page-marker) .stTextInput > label {
    color: #334155 !important; font-weight: 600 !important;
    font-size: 0.85rem !important; margin-bottom: 0.4rem !important;
}
.stApp:has(.login-page-marker) .stTextInput input {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    color: #1e293b !important; border-radius: 8px !important;
    padding: 0.7rem 1rem !important; font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}
.stApp:has(.login-page-marker) .stTextInput input:focus {
    background: #ffffff !important; border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important; outline: none !important;
}
.stApp:has(.login-page-marker) .stTextInput input::placeholder { color: #94a3b8 !important; }

/* Button */
.stApp:has(.login-page-marker) .stFormSubmitButton button {
    padding: 0.75rem 1.5rem !important; font-size: 0.95rem !important;
    font-weight: 600 !important; border-radius: 8px !important;
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important; color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.2s ease !important; cursor: pointer;
    margin-top: 0.5rem !important;
}
.stApp:has(.login-page-marker) .stFormSubmitButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
}

/* Demo box */
.lc-demo {
    text-align: center; padding: 0.75rem; margin-top: 1rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0; border-radius: 8px;
}
.lc-demo p { color: #64748b; font-size: 0.85rem; margin: 0; line-height: 1.5; }
.lc-demo strong {
    color: #3b82f6; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    background: #dbeafe; padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* Alerts */
.stApp:has(.login-page-marker) .stAlert { border-radius: 8px !important; }

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


def get_custom_css() -> str:
    """
    Return the custom CSS as a string (for programmatic access).
    
    This function is useful if you need to:
      - Export CSS to a file
      - Inject CSS conditionally
      - Combine with other CSS sources
    
    Returns:
        str: Complete CSS stylesheet as a string
    """
    return """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

/* Original styling */
html, body, [class*="css"] { 
    font-family: 'IBM Plex Sans', sans-serif; 
}
code, pre { 
    font-family: 'IBM Plex Mono', monospace !important; 
}

/* Original topbar */
.topbar { 
    background: linear-gradient(90deg, #0f172a, #1e3a5f); 
    color: white; 
    padding: 18px 28px; 
    border-radius: 12px; 
    margin-bottom: 20px; 
}
.topbar h1 { 
    margin: 0; 
    font-size: 1.4rem; 
    font-weight: 700; 
}
.topbar p { 
    margin: 0; 
    font-size: 0.82rem; 
    color: #94a3b8; 
}

/* Original KPI cards */
.kpi { 
    background: #0f172a; 
    border: 1px solid #1e3a5f; 
    border-radius: 10px; 
    padding: 16px; 
    text-align: center; 
}
.kpi-val { 
    font-size: 1.8rem; 
    font-weight: 700; 
    color: #38bdf8; 
}
.kpi-lbl { 
    font-size: 0.78rem; 
    color: #64748b; 
    margin-top: 4px; 
}

/* Original log box */
.log-box { 
    background: #0f172a; 
    border: 1px solid #1e3a5f; 
    border-radius: 8px; 
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace; 
    font-size: 0.78rem; 
    color: #94a3b8;
    max-height: 320px; 
    overflow-y: auto; 
}

/* Original mode card */
.mode-card { 
    background: #0f172a; 
    border: 1px solid #334155; 
    border-radius: 10px; 
    padding: 14px 18px; 
    margin: 6px 0; 
}
.mode-card h4 { 
    margin: 0 0 4px; 
    color: #38bdf8; 
    font-size: 0.95rem; 
}
.mode-card p { 
    margin: 0; 
    font-size: 0.82rem; 
    color: #94a3b8; 
}

/* Sidebar - fixed collapse behavior */
section[data-testid="stSidebar"] {
    min-width: 80px !important;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 280px !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 80px !important;
}

/* Sidebar buttons - original colors */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: #94a3b8;
    border: 1px solid #1e293b;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.25s ease;
    padding: 9px 14px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border-color: #38bdf8;
}

/* ===== LOGIN PAGE ===== */

/* Hide sidebar and header when login-page marker is present */
.stApp:has(.login-page-marker) header[data-testid="stHeader"],
.stApp:has(.login-page-marker) section[data-testid="stSidebar"],
.stApp:has(.login-page-marker) .stDeployButton {
    display: none !important;
}

/* Light gradient background */
.stApp:has(.login-page-marker) {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #f8fafc 100%) !important;
}

/* Center everything */
.stApp:has(.login-page-marker) .main,
.stApp:has(.login-page-marker) [data-testid="stAppViewContainer"],
.stApp:has(.login-page-marker) section[data-testid="stMain"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 100vh !important;
}

.stApp:has(.login-page-marker) .block-container,
.stApp:has(.login-page-marker) [data-testid="block-container"] {
    padding: 2rem !important;
    max-width: 450px !important;
    width: 100% !important;
}

/* Clean white card */
.stApp:has(.login-page-marker) [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 2.5rem 2rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    animation: cardSlideUp 0.5s ease-out;
    width: 100% !important;
}

@keyframes cardSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Seamless form */
.stApp:has(.login-page-marker) [data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    box-shadow: none !important;
}

/* Brand section */
.lc-brand { text-align: center; margin-bottom: 1.5rem; }
.lc-icon {
    width: 56px; height: 56px;
    margin: 0 auto 1rem;
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
}
.lc-icon i { font-size: 1.6rem; color: #ffffff; }
.lc-brand h2 {
    color: #1e293b; font-size: 1.5rem; font-weight: 700;
    margin: 0 0 0.3rem; letter-spacing: -0.02em;
}
.lc-brand p { color: #64748b; font-size: 0.9rem; margin: 0; }

/* Input styling */
.stApp:has(.login-page-marker) .stTextInput > label {
    color: #334155 !important; font-weight: 600 !important;
    font-size: 0.85rem !important; margin-bottom: 0.4rem !important;
}
.stApp:has(.login-page-marker) .stTextInput input {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    color: #1e293b !important; border-radius: 8px !important;
    padding: 0.7rem 1rem !important; font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}
.stApp:has(.login-page-marker) .stTextInput input:focus {
    background: #ffffff !important; border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important; outline: none !important;
}
.stApp:has(.login-page-marker) .stTextInput input::placeholder { color: #94a3b8 !important; }

/* Button */
.stApp:has(.login-page-marker) .stFormSubmitButton button {
    padding: 0.75rem 1.5rem !important; font-size: 0.95rem !important;
    font-weight: 600 !important; border-radius: 8px !important;
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important; color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.2s ease !important; cursor: pointer;
    margin-top: 0.5rem !important;
}
.stApp:has(.login-page-marker) .stFormSubmitButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
}

/* Demo box */
.lc-demo {
    text-align: center; padding: 0.75rem; margin-top: 1rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0; border-radius: 8px;
}
.lc-demo p { color: #64748b; font-size: 0.85rem; margin: 0; line-height: 1.5; }
.lc-demo strong {
    color: #3b82f6; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    background: #dbeafe; padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* Alerts */
.stApp:has(.login-page-marker) .stAlert { border-radius: 8px !important; }

/* Remove Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
"""
