/* ============================================================
   EnsetScan Vision — Application Logic
   Offline-first plant disease diagnosis:
   • TensorFlow.js pipeline (real model) with tf.tidy() memory guards
   • Heuristic fallback classifier (deterministic, camera-safe)
   • 65% Confidence Safeguard Threshold
   • IndexedDB text-only audit log (no image data persisted)
   ============================================================ */

'use strict';

/* ---------------- Constants ---------------- */
const CONFIDENCE_THRESHOLD = 0.65; // 65% safeguard
const MODEL_URL = 'model.json';
const CATALOG_URL = 'disease-catalog.json';
const INPUT_SIZE = 224;             // neural input 224x224
const HEURISTIC_SIZE = 96;          // heuristic working resolution
const DB_NAME = 'ensetscan-db';
const DB_VERSION = 1;
const DB_STORE = 'scans';

/* ---------------- Global State ---------------- */
let catalog = null;                 // disease knowledge base
let model = null;                   // tf.LayersModel (if loaded)
let modelStatus = 'loading';        // 'neural' | 'heuristic' | 'unavailable'
let currentImage = null;            // HTMLImageElement held in RAM only
let currentObjectUrl = null;        // blob URL to revoke after processing
let db = null;                      // IndexedDB database handle
let deferredInstallPrompt = null;   // captured beforeinstallprompt event
let lang = 'en';                    // current UI language: 'en' | 'am'
let lastResult = null;              // last rendered result (for language re-render)
let hashRouteReady = false;         // ignore initial-load hashchange events
let dashData = null;                // national dashboard knowledge base
let dashTab = 'threats';            // active dashboard tab
let cameraStream = null;            // active getUserMedia stream
let cameraActive = false;           // whether the viewfinder is live
let cameraStable = false;           // stability detector state
let cameraStableSince = 0;          // timestamp when stability was achieved
let cameraLight = 'checking';       // 'good' | 'dark' | 'bright' | 'checking'
let cameraRaf = null;               // requestAnimationFrame handle
let cameraPrevFrame = null;         // previous downscaled frame for motion detection
let cameraMotion = 0;               // current motion score 0..1
let cameraSupported = false;        // whether getUserMedia is available
let mlWorker = null;                // dedicated ML worker (heuristic offload)
let mlWorkerReady = false;          // whether the worker responded successfully

/* ---------------- DOM References ---------------- */
const el = (id) => document.getElementById(id);
const cameraCard = el('cameraCard');
const cameraVideo = el('cameraVideo');
const cameraFrame = el('cameraFrame');
const reticle = el('reticle');
const cameraStatus = el('cameraStatus');
const lightChip = el('lightChip');
const lightChipText = el('lightChipText');
const stabilityChip = el('stabilityChip');
const stabilityChipText = el('stabilityChipText');
const cameraError = el('cameraError');
const cameraErrorText = el('cameraErrorText');
const captureBtn = el('captureBtn');
const captureBtnText = el('captureBtnText');
const galleryBtn = el('galleryBtn');
const galleryBtnText = el('galleryBtnText');
const chooser = el('chooser');
const chooserCameraBtn = el('chooserCameraBtn');
const chooserGalleryBtn = el('chooserGalleryBtn');
const chooserCancelBtn = el('chooserCancelBtn');
const chooserTitle = el('chooserTitle');
const chooserCameraText = el('chooserCameraText');
const chooserGalleryText = el('chooserGalleryText');
const startScan = el('startScan');
const startScanBtn = el('startScanBtn');
const previewCard = el('previewCard');
const loadingCard = el('loadingCard');
const resultCard = el('resultCard');
const previewImg = el('previewImg');
const photoInput = el('photoInput');
const loadingText = el('loadingText');
const resultIcon = el('resultIcon');
const resultTitle = el('resultTitle');
const resultSubtitle = el('resultSubtitle');
const confidenceValue = el('confidenceValue');
const confidenceFill = el('confidenceFill');
const visualMatch = el('visualMatch');
const adviceList = el('adviceList');
const historyList = el('historyList');
const historyEmpty = el('historyEmpty');
const installBtn = el('installBtn');
const installBanner = el('installBanner');
const installBannerBtn = el('installBannerBtn');
const langEnBtn = el('langEnBtn');
const langAmBtn = el('langAmBtn');
const installInfoCard = el('installInfoCard');
const scanCountEl = el('scanCount');
const lastScanEl = el('lastScan');
const engineBadgeEl = el('engineBadge');
const engineBadgeText = el('engineBadgeText');
const dashBody = el('dashBody');
const viewHome = el('view-home');
const viewDashboard = el('view-dashboard');
const viewHistory = el('view-history');
const viewTitle = el('viewTitle');

/* ---------------- UI Translation Strings ---------------- */
const I18N = {
  en: {
    view_home: 'New Scan',
    view_dashboard: 'Disease Dashboard',
    view_history: 'Scan History',
    disease: 'DETECTED: ',
    inconclusive: 'Inconclusive Scan',
    inconclusive_desc: 'Below the 65% confidence safeguard threshold · Engine: ',
    high_certainty: 'High Certainty',
    confirmed: 'Confirmed Match (≥65%)',
    visual_match: '🔍 Visual Match',
    local_names: '🌐 Local Names',
    farmer_action: '💡 Recommended Farmer Action',
    landrace_title: '🌱 Landrace Recommendation',
    landrace_resistant: 'Plant resistant landraces: ',
    landrace_susceptible: 'Avoid susceptible landraces: ',
    food_loss_title: '🍲 Enset Food Loss Estimator',
    food_loss_note: 'Estimated household Kocho and Bulla food reserve loss per infected plant.',
    food_loss_per: 'per infected plant',
    food_loss_plants: 'infected plant(s)',
    food_loss_total: 'Estimated total loss',
    food_loss_kocho: 'kg Kocho',
    food_loss_bulla: 'kg Bulla',
    canopy_title: '☀️ Intercropping Canopy Advisor',
    sanitation_title: '🔪 Tool Sanitation Priority',
    regional_title: '📍 Regional Context',
    traditional_title: '🧺 Traditional Practices',
    resistant_accessions: '🌿 Resistant Accessions',
    analyze: 'Analyzing leaf…',
    install_title: '📲 Install as Android App',
    install_body: 'Open this site in Chrome, tap the menu (⋮) and choose "Install app" or "Add to Home screen" to use EnsetScan like a native app — fully offline.',
    install_go: 'Install',
    history_empty: 'No scans recorded yet.',
    clear_history: '🗑 Clear History',
    engine: 'Engine',
    hero_title: 'Diagnose your crops in seconds',
    hero_body: 'Snap or upload a photo of a leaf from Enset, Coffee, or Maize. EnsetScan Vision runs a machine-learning model entirely on your device — no internet, no data cost, no upload.',
    upload_title: 'Take a photo or choose a leaf image',
    upload_hint: 'Tap to open your camera or file picker',
    privacy_note: '🔒 Your photo is processed in temporary memory and never saved or uploaded.',
    analyze_btn: '🔍 Analyze Leaf',
    choose_btn: '↺ Choose Another',
    new_scan_btn: '🌿 New Scan',
    history_btn: 'History',
    history_title: '🕘 Scan History',
    preview_title: '📷 Photo preview',
    cancel_aria: 'Cancel scan',
    confidence_label: 'Confidence',
    visual_match_title: '🔍 Visual Match',
    farmer_action_title: '💡 Recommended Farmer Action',
    below_safeguard: 'Below Safeguard',
    footer: '🌿 EnsetScan Vision · Offline-first · Zero data cost · Works without internet',
    install_btn: 'Install',
    home_remedies_title: '🏠 Traditional & Easy Home Remedies',
    seasonality_title: '📅 Seasonality',
    transmission_title: '🔄 Transmission',
    spread_rate_title: '⚡ Spread Rate',
    yield_impact_title: '📉 Yield Impact',
    monitoring_title: '🔍 Monitoring',
    severity_title: '⭐ Severity Scale',
    prevention_title: '🛡️ Prevention Tips',
    spread_fast: 'Fast',
    spread_moderate: 'Moderate',
    spread_slow: 'Slow',
    spread_none: 'None',
    scans_done: 'Scans done',
    last_scan: 'Last scan',
    engine_badge: 'Engine',
    capture_btn: 'Capture Leaf',
    gallery_btn: 'Choose from Gallery',
    light_checking: 'Checking light…',
    light_good: 'Good lighting',
    light_dark: 'Too dark',
    light_bright: 'Too bright',
    steady_hold: 'Hold camera steady',
    steady_ok: 'Camera steady',
    chooser_title: 'Add a leaf photo',
    chooser_camera: 'Take Photo',
    chooser_gallery: 'Choose from Gallery',
    chooser_cancel: 'Cancel',
    start_scan_title: 'Ready to diagnose?',
    start_scan_body: 'Take a photo or pick one from your gallery to get started.',
    start_scan_btn: 'Start Scan'
  },
  am: {
    view_home: 'አዲስ ምርመራ',
    view_dashboard: 'የበሽታ ዳሽቦርድ',
    view_history: 'የምርመራ ታሪክ',
    disease: 'ተገኝቷል፡ ',
    inconclusive: 'ያልተረጋገጠ ምርመራ',
    inconclusive_desc: 'ከ65% የመተማመን ደረጃ በታች · ሞተር፡ ',
    high_certainty: 'ከፍተኛ እርግጠኝነት',
    confirmed: 'የተረጋገጠ ውጤት (≥65%)',
    visual_match: '🔍 የእይታ ንጽጽር',
    local_names: '🌐 የአካባቢ ስሞች',
    farmer_action: '💡 የሚመከር የገበሬ እርምጃ',
    landrace_title: '🌱 የእንሰት ዝርያ ምክር',
    landrace_resistant: 'ተከላካይ ዝርያዎችን ይትከሉ፡ ',
    landrace_susceptible: 'ተጋላጭ ዝርያዎችን ያስወግዱ፡ ',
    food_loss_title: '🍲 የእንሰት ምግብ ኪሳራ ግምት',
    food_loss_note: 'በአንድ የተያዘ ተክል የሚጠፋ የኩቾ እና የቡላ የቤት ምግብ ክምችት ግምት።',
    food_loss_per: 'በአንድ የተያዘ ተክል',
    food_loss_plants: 'የተያዙ ተክሎች',
    food_loss_total: 'ጠቅላላ የሚጠፋ ክምችት',
    food_loss_kocho: 'ኪግ ኩቾ',
    food_loss_bulla: 'ኪግ ቡላ',
    canopy_title: '☀️ የተጣመረ የጥላ ምክር',
    sanitation_title: '🔪 የመሳሪያ ንጽህና ቅድሚያ',
    regional_title: '📍 የአካባቢ ሁኔታ',
    traditional_title: '🧺 ባህላዊ ልምዶች',
    resistant_accessions: '🌿 ተከላካይ ዝርያዎች',
    analyze: 'ቅጠሉ እየተመረመረ ነው…',
    install_title: '📲 እንደ አንድሮይድ መተግበሪያ ይጫኑ',
    install_body: 'ይህን ጣቢያ በChrome ይክፈቱ፣ ሜኑውን (⋮) ይንኩ እና "መተግበሪያ ጫን" ወይም "ወደ መነሻ ማያ ገጽ ጨምር" ይምረጡ።',
    install_go: 'ጫን',
    history_empty: 'እስካሁን ምንም ምርመራ አልተመዘገበም።',
    clear_history: '🗑 ታሪክ አጽዳ',
    engine: 'ሞተር',
    hero_title: 'ሰብሎችዎን በሰከንዶች ውስጥ ይመርምሩ',
    hero_body: 'ከእንሰት፣ ቡና ወይም በቆሎ የቅጠል ፎቶ ያንሱ ወይም ይምረጡ። EnsetScan Vision የማሽን መማሪያ ሞዴልን ሙሉ በሙሉ በመሳሪያዎ ላይ ያሂዳል — ያለ ኢንተርኔት፣ ያለ የውሂብ ወጪ፣ ያለ መጫን።',
    upload_title: 'የቅጠል ፎቶ ያንሱ ወይም ይምረጡ',
    upload_hint: 'ካሜራዎን ወይም የፋይል መራጩን ለመክፈት ይንኩ',
    privacy_note: '🔒 ፎቶዎ በጊዜያዊ ማህደረ-ትውስታ ውስጥ ይሰራል እንጂ አይቀመጥም ወይም አይጫንም።',
    analyze_btn: '🔍 ቅጠሉን ይመርምሩ',
    choose_btn: '↺ ሌላ ይምረጡ',
    new_scan_btn: '🌿 አዲስ ምርመራ',
    history_btn: 'ታሪክ',
    history_title: '🕘 የምርመራ ታሪክ',
    preview_title: '📷 የፎቶ ቅድመ-እይታ',
    cancel_aria: 'ምርመራ ሰርዝ',
    confidence_label: 'እርግጠኝነት',
    visual_match_title: '🔍 የእይታ ንጽጽር',
    farmer_action_title: '💡 የሚመከር የገበሬ እርምጃ',
    below_safeguard: 'ከመተማመን ደረጃ በታች',
    footer: '🌿 EnsetScan Vision · ከመስመር ውጭ · ዜሮ የውሂብ ወጪ · ያለ ኢንተርኔት ይሰራል',
    install_btn: 'ጫን',
    home_remedies_title: '🏠 ባህላዊ እና ቀላል የቤት መፍትሄዎች',
    seasonality_title: '📅 ወቅት',
    transmission_title: '🔄 ስርጭት',
    spread_rate_title: '⚡ የስርጭት ፍጥነት',
    yield_impact_title: '📉 የምርት ኪሳራ',
    monitoring_title: '🔍 ክትትል',
    severity_title: '⭐ የከባድነት መጠን',
    prevention_title: '🛡️ የመከላከያ ምክሮች',
    spread_fast: 'ፈጣን',
    spread_moderate: 'መካከለኛ',
    spread_slow: 'ቀስቃሽ',
    spread_none: '—',
    scans_done: 'የተደረጉ ምርመራዎች',
    last_scan: 'የመጨረሻ ምርመራ',
    engine_badge: 'ሞተር',
    capture_btn: 'ቅጠሉን ይያዙ',
    gallery_btn: 'ከማህደር ይምረጡ',
    light_checking: 'ብርሀን በመፈተሽ ላይ…',
    light_good: 'ብርሀኑ ጥሩ ነው',
    light_dark: 'በጣም ጨለማ',
    light_bright: 'በጣም ብሩህ',
    steady_hold: 'ካሜራውን በእጅዎ ይያዙ',
    steady_ok: 'ካሜራው ተረጋግቷል',
    chooser_title: 'የቅጠል ፎቶ ያክሉ',
    chooser_camera: 'ፎቶ ያንሱ',
    chooser_gallery: 'ከማህደር ይምረጡ',
    chooser_cancel: 'ሰርዝ',
    start_scan_title: 'ለመመርመር ዝግጁ ነዎት?',
    start_scan_body: 'ፎቶ ያንሱ ወይም ከጋለሪዎ ይምረጡ ለመጀመር።',
    start_scan_btn: 'ምርመራ ይጀምሩ'
  }
};
let lastOpenMode = 'camera'; // 'camera' | 'gallery'
let cameraStarting = false; // guard against re-entry
let cameraStale = false;

function bindEvents() {
  photoInput.addEventListener('change', handleFileSelect);

  el('analyzeBtn').addEventListener('click', runDiagnosis);
  el('retakeBtn').addEventListener('click', resetToUpload);
  el('newScanBtn').addEventListener('click', resetToUpload);
  const cancelScanBtn = el('cancelScanBtn');
  if (cancelScanBtn) cancelScanBtn.addEventListener('click', resetToUpload);

  el('clearHistoryBtn').addEventListener('click', clearHistory);

  // Main view navigation (Home / Dashboard / History)
  document.querySelectorAll('.nav-item').forEach((item) => {
    item.addEventListener('click', () => switchView(item.dataset.view));
  });

  // Dashboard tabs
  document.querySelectorAll('.dash-tab').forEach((tab) => {
    tab.addEventListener('click', () => switchDashTab(tab.dataset.tab));
  });

  // Start Scan button → opens the photo source chooser
  if (startScanBtn) startScanBtn.addEventListener('click', () => {
    hide(startScan);
    show(cameraCard);
    openChooser();
  });

  // Camera viewfinder controls: tapping the photo box shows the chooser.
  // Only trigger when clicking the frame area, NOT the buttons inside.
  if (cameraCard) cameraCard.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    openChooser();
  });
  if (chooser) chooser.addEventListener('click', closeChooserOnBackdrop);

  // Photo source chooser options
  if (chooserCameraBtn) chooserCameraBtn.addEventListener('click', () => { hide(chooser); startCamera(); });
  if (chooserGalleryBtn) chooserGalleryBtn.addEventListener('click', () => { hide(chooser); photoInput.click(); });
  if (chooserCancelBtn) chooserCancelBtn.addEventListener('click', () => hide(chooser));

  // Camera viewfinder controls
  if (captureBtn) captureBtn.addEventListener('click', captureFrame);
  if (galleryBtn) galleryBtn.addEventListener('click', () => photoInput.click());

  // PWA install button (Chrome/Android only — hidden until prompt fires)
  if (installBtn) installBtn.addEventListener('click', installApp);
  if (installBannerBtn) installBannerBtn.addEventListener('click', installApp);

  // Language selection (English / Amharic)
  if (langEnBtn) langEnBtn.addEventListener('click', () => setLang('en'));
  if (langAmBtn) langAmBtn.addEventListener('click', () => setLang('am'));

  // Install info card button (launches prompt if available)
  if (installInfoCard) {
    const goBtn = installInfoCard.querySelector('[data-i18n="install_go"]');
    if (goBtn) goBtn.addEventListener('click', () => installBtn && installBtn.click());
  }

  // Hash-route shortcuts declared in manifest.webmanifest.
  // Only respond to hash *changes* (e.g. launcher shortcuts), and ignore the
  // initial-load hashchange so a leftover #history can't re-open the modal.
  window.addEventListener('hashchange', handleHashRoute);
}

function openChooser() {
  if (!chooser) return;
  const t = I18N[lang];
  if (chooserTitle) chooserTitle.innerHTML = '<svg class="icon icon-sm"><use href="#i-camera"/></svg> ' + t.chooser_title;
  if (chooserCameraText) chooserCameraText.textContent = t.chooser_camera;
  if (chooserGalleryText) chooserGalleryText.textContent = t.chooser_gallery;
  if (chooserCancelBtn) chooserCancelBtn.textContent = '✕ ' + t.chooser_cancel;
  show(chooser);
}

function closeChooserOnBackdrop(e) {
  if (e.target === chooser) hide(chooser);
}

/* ============================================================
   INITIALIZATION
   ============================================================ */
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch(CATALOG_URL);
    catalog = await res.json();
  } catch (err) {
    console.error('Failed to load disease catalog:', err);
  }

  // Load saved language preference
  lang = (typeof localStorage !== 'undefined' && localStorage.getItem('ensetscan-lang')) || 'en';
  applyLang();

  // Load national dashboard data (non-blocking)
  await loadDashData();

  await initDatabase();
  bindEvents();
  registerServiceWorker();
  initMLWorker(); // offload heuristic inference to a worker when available
  initModel(); // fire-and-forget; heuristic engages if model unavailable
  updateAppStats();

  // Always open on the home/scan view, regardless of any leftover hash,
  // launcher shortcut, or stale cache state.
  forceHomeView();

  // The browser fires an initial `hashchange` event on page load if the URL
  // carries a hash (e.g. a leftover #history). That would re-open the modal
  // right after forceHomeView() closed it. So we ignore hash changes during
  // the first 500ms of load; later changes (PWA launcher shortcuts) still work.
  setTimeout(() => { hashRouteReady = true; }, 500);
});

/** Update the footer stats (scan count + last scan) and engine badge. */
function updateAppStats(scanTimestamp) {
  const t = I18N[lang];

  // Load / increment the persistent scan counter
  let scans = 0;
  try { scans = parseInt(localStorage.getItem('ensetscan-scans') || '0', 10); } catch (e) { }
  if (scanTimestamp) {
    scans += 1;
    try { localStorage.setItem('ensetscan-scans', String(scans)); } catch (e) { }
    try { localStorage.setItem('ensetscan-lastscan', scanTimestamp); } catch (e) { }
  } else {
    // On initial load, count existing records in IndexedDB
    getAllScans().then((records) => {
      if (records.length > scans) {
        try { localStorage.setItem('ensetscan-scans', String(records.length)); } catch (e) { }
        scans = records.length;
      }
      if (scanCountEl) {
        scanCountEl.textContent =
          t.scans_done + ': ' + scans;
      }
    });
  }

  if (scanCountEl) scanCountEl.textContent = t.scans_done + ': ' + scans;

  // Last scan timestamp
  let lastTs = null;
  try { lastTs = localStorage.getItem('ensetscan-lastscan'); } catch (e) { }
  if (lastScanEl) {
    lastScanEl.textContent = lastTs
      ? t.last_scan + ': ' + new Date(lastTs).toLocaleString()
      : t.last_scan + ': —';
  }

  // Engine badge
  if (engineBadgeText) {
    engineBadgeText.textContent =
      t.engine_badge + ': ' + (modelStatus === 'neural' ? 'Neural' : 'Heuristic');
  }
}

/** Force the app to the home/scan view on every page load. */
function forceHomeView() {
  // Ensure the home view is the active one
  switchView('home');
  // Clear a leftover #history hash so it can't re-open the view.
  // Wrap in try/catch: replaceState can throw a SecurityError on file:// URLs.
  if (window.location.hash === '#history') {
    try {
      history.replaceState(null, '', window.location.pathname + window.location.search);
    } catch (e) {
      // Ignore — the view is already on home; the hash is cosmetic.
    }
  }
  // Show the welcome/splash card and hide all workflow cards
  hide(cameraCard);
  hide(previewCard);
  hide(loadingCard);
  hide(resultCard);
  show(startScan);
  // Reset any lingering result state
  lastResult = null;
  // Ensure any stale camera streams are cleaned up
  stopCamera();
}

/* ---------------- Language Selection ---------------- */
function setLang(newLang) {
  if (newLang !== 'en' && newLang !== 'am') return;
  lang = newLang;
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem('ensetscan-lang', lang);
  }
  applyLang();
}

/** Apply language to all static UI strings (page-level, pre-result text). */
function applyLang() {
  const t = I18N[lang];
  // Highlight the active language button
  if (langEnBtn) langEnBtn.classList.toggle('active', lang === 'en');
  if (langAmBtn) langAmBtn.classList.toggle('active', lang === 'am');
  loadingText.textContent = t.analyze;

  // Toggle Ethiopic font stack + line-height for Amharic
  document.documentElement.lang = lang;
  document.body.classList.toggle('lang-am', lang === 'am');

  // Map data-i18n attribute → translation key for all static elements
  const i18nMap = {
    heroTitle: t.hero_title,
    heroBody: t.hero_body,
    uploadTitle: t.upload_title,
    uploadHint: t.upload_hint,
    privacyNote: t.privacy_note,
    analyzeBtn: t.analyze_btn,
    chooseBtn: t.choose_btn,
    newScanBtn: t.new_scan_btn,
    historyBtn: t.history_btn,
    historyTitle: t.history_title,
    previewTitle: t.preview_title,
    cancelAria: t.cancel_aria,
    confidenceLabel: t.confidence_label,
    visualMatchTitle: t.visual_match_title,
    farmerActionTitle: t.farmer_action_title,
    footer: t.footer,
    installBtn: t.install_btn,
    captureBtnText: t.capture_btn,
    galleryBtnText: t.gallery_btn,
    clearHistoryBtnText: t.clear_history,
    startScanTitle: t.start_scan_title,
    startScanBody: t.start_scan_body,
    startScanBtnText: t.start_scan_btn
  };
  Object.keys(i18nMap).forEach((id) => {
    const node = document.getElementById(id);
    if (node) node.textContent = i18nMap[id];
  });

  // The install info card translations
  if (installInfoCard) {
    const elt = installInfoCard.querySelector('[data-i18n="install_title"]');
    if (elt) elt.textContent = t.install_title;
    const elb = installInfoCard.querySelector('[data-i18n="install_body"]');
    if (elb) elb.textContent = t.install_body;
  }

  // Re-render active result card in the new language
  if (lastResult) {
    if (lastResult.inconclusive) {
      renderInconclusive(lastResult.conf, lastResult.engine);
    } else {
      renderResult(lastResult.cls, lastResult.conf, lastResult.engine);
    }
  }
}

/** Localized helper: pick the Amharic or English value from a class field. */
function loc(cls, enKey, amKey) {
  if (lang === 'am' && cls[amKey] !== undefined) return cls[amKey];
  return cls[enKey];
}

function handleHashRoute() {
  // History is opened ONLY via the History button — never from a URL hash.
  // This guarantees the modal can never auto-open on page load, regardless
  // of any leftover #history hash, cache state, or launcher shortcut.
  // The #scan shortcut simply resets to the upload view (harmless).
  if (window.location.hash === '#scan') {
    resetToUpload();
    if (window.location.hash === '#scan') history.replaceState(null, '', '#scan');
  }
}

/* ---------------- PWA Install & Hash Routes ---------------- */
function installApp() {
  if (!deferredInstallPrompt) {
    // No install prompt available (e.g. already installed or unsupported).
    // Keep the banner visible but do nothing harmful.
    console.info('Install prompt not available.');
    return;
  }
  deferredInstallPrompt.prompt();
  deferredInstallPrompt.userChoice.then(() => {
    deferredInstallPrompt = null;
    installBtn.hidden = true;
    if (installBanner) installBanner.hidden = true;
  });
}

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredInstallPrompt = e;
  installBtn.hidden = false;
  if (installBanner) installBanner.hidden = false;
});

window.addEventListener('appinstalled', () => {
  deferredInstallPrompt = null;
  installBtn.hidden = true;
  if (installBanner) installBanner.hidden = true;
});

/* ---------------- Service Worker ---------------- */
function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('sw.js').catch((err) => {
        console.warn('Service worker registration failed:', err);
      });
    });
  }
}

/* ============================================================
   LIVE CAMERA VIEWFINDER — getUserMedia + light/stability checks
   ============================================================ */

/** Haptic feedback helper — guarded for unsupported devices. */
function vibrate(pattern) {
  if (typeof navigator !== 'undefined' && 'vibrate' in navigator) {
    try { navigator.vibrate(pattern); } catch (e) { /* ignore */ }
  }
}

/** Start the live camera viewfinder (or fall back to gallery). */
async function startCamera() {
  if (!cameraCard || !cameraVideo) return;
  if (cameraActive) return; // already running

  cameraSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  if (!cameraSupported) {
    showCameraError();
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    cameraVideo.srcObject = cameraStream;
    cameraActive = true;
    cameraError.hidden = true;
    cameraVideo.hidden = false;
    reticle.hidden = false;
    cameraStatus.hidden = false;
    captureBtn.disabled = true;
    // Reset detector state
    cameraStable = false;
    cameraStableSince = 0;
    cameraLight = 'checking';
    cameraPrevFrame = null;
    cameraMotion = 0;
    updateLightChip();
    updateStabilityChip();
    // Start the analysis loop
    if (cameraRaf) cancelAnimationFrame(cameraRaf);
    cameraRaf = requestAnimationFrame(cameraLoop);
  } catch (err) {
    console.warn('Camera unavailable:', err.message);
    showCameraError();
  }
}

/** Stop the live camera stream and release resources. */
function stopCamera() {
  if (cameraRaf) {
    cancelAnimationFrame(cameraRaf);
    cameraRaf = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }
  cameraActive = false;
  cameraPrevFrame = null;
  if (cameraVideo) cameraVideo.srcObject = null;
}

/** Show the camera-unavailable fallback UI. */
function showCameraError() {
  if (!cameraError) return;
  cameraError.hidden = false;
  if (cameraVideo) cameraVideo.hidden = true;
  if (reticle) reticle.hidden = true;
  if (cameraStatus) cameraStatus.hidden = true;
  if (captureBtn) captureBtn.disabled = true;
}

/** Main rAF loop: sample light + motion, update chips and reticle. */
function cameraLoop() {
  if (!cameraActive || !cameraVideo || cameraVideo.readyState < 2) {
    cameraRaf = requestAnimationFrame(cameraLoop);
    return;
  }

  // Downscale to a tiny working canvas for cheap pixel analysis
  const w = 64, h = 48;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(cameraVideo, 0, 0, w, h);
  const frame = ctx.getImageData(0, 0, w, h).data;

  // --- Ambient light check (average luminance) ---
  let lumSum = 0;
  for (let i = 0; i < frame.length; i += 4) {
    lumSum += 0.299 * frame[i] + 0.587 * frame[i + 1] + 0.114 * frame[i + 2];
  }
  const avgLum = lumSum / (frame.length / 4);
  const newLight = avgLum < 45 ? 'dark' : (avgLum > 215 ? 'bright' : 'good');
  if (newLight !== cameraLight) {
    cameraLight = newLight;
    updateLightChip();
  }

  // --- Stability / motion detection (frame difference) ---
  if (cameraPrevFrame) {
    let diff = 0;
    let count = 0;
    for (let i = 0; i < frame.length; i += 4) {
      const d = Math.abs(frame[i] - cameraPrevFrame[i]) +
                Math.abs(frame[i + 1] - cameraPrevFrame[i + 1]) +
                Math.abs(frame[i + 2] - cameraPrevFrame[i + 2]);
      diff += d;
      count++;
    }
    cameraMotion = Math.min(1, diff / (count * 3 * 60)); // normalize
  }
  cameraPrevFrame = frame;

  const now = performance.now();
  if (cameraMotion < 0.12) {
    if (!cameraStable) {
      cameraStable = true;
      cameraStableSince = now;
    }
  } else {
    cameraStable = false;
    cameraStableSince = 0;
  }

  // Enable capture only when stable for ~600ms AND lighting is good
  const stableEnough = cameraStable && (now - cameraStableSince) >= 600;
  const lightOk = cameraLight === 'good';
  captureBtn.disabled = !(stableEnough && lightOk);

  // Update reticle state (stable pulses green, motion pulses amber)
  if (reticle) {
    reticle.classList.toggle('stable', stableEnough);
    reticle.classList.toggle('motion', !stableEnough && cameraMotion >= 0.12);
  }

  updateStabilityChip();

  cameraRaf = requestAnimationFrame(cameraLoop);
}

/** Update the light status chip. */
function updateLightChip() {
  if (!lightChip || !lightChipText) return;
  const t = I18N[lang];
  lightChip.className = 'status-chip status-' + cameraLight;
  lightChipText.textContent =
    cameraLight === 'good' ? t.light_good :
    cameraLight === 'dark' ? t.light_dark :
    cameraLight === 'bright' ? t.light_bright : t.light_checking;
}

/** Update the stability status chip. */
function updateStabilityChip() {
  if (!stabilityChip || !stabilityChipText) return;
  const t = I18N[lang];
  const stableEnough = cameraStable && (performance.now() - cameraStableSince) >= 600;
  stabilityChip.className = 'status-chip ' + (stableEnough ? 'status-good' : 'status-warn');
  stabilityChipText.textContent = stableEnough ? t.steady_ok : t.steady_hold;
}

/** Capture the current video frame to a Blob and feed the pipeline. */
function captureFrame() {
  if (!cameraActive || !cameraVideo || captureBtn.disabled) return;

  vibrate([15, 30]); // haptic on capture

  const canvas = document.createElement('canvas');
  canvas.width = cameraVideo.videoWidth || 640;
  canvas.height = cameraVideo.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    if (!blob) return;
    // Stop the camera stream to save battery
    stopCamera();

    // Build a File-like object and reuse the existing file pipeline
    const file = new File([blob], 'leaf-capture.jpg', { type: 'image/jpeg' });
    const fakeEvent = { target: { files: [file] } };
    handleFileSelect(fakeEvent);
  }, 'image/jpeg', 0.92);
}

/* ============================================================
   ML WORKER — offload heuristic inference off the main thread
   ============================================================ */
function initMLWorker() {
  if (typeof Worker === 'undefined') return;
  try {
    mlWorker = new Worker('worker.js');
    mlWorker.onmessage = (e) => {
      if (e.data && e.data.type === 'result') {
        mlWorkerReady = true;
      }
    };
    mlWorker.onerror = () => {
      // Worker failed to load/run — fall back to main-thread inference.
      mlWorker = null;
      mlWorkerReady = false;
    };
  } catch (err) {
    mlWorker = null;
    mlWorkerReady = false;
  }
}

/**
 * Run inference, preferring the dedicated worker when available.
 * Falls back to main-thread heuristic if the worker is unavailable
 * or fails to respond.
 */
function runInference(img) {
  return new Promise((resolve) => {
    const fallback = () => {
      const r = analyzeHeuristic(img);
      resolve({ probs: r.probs, engine: 'heuristic', elapsedMs: r.elapsedMs });
    };

    if (!mlWorker || !mlWorkerReady) {
      fallback();
      return;
    }

    // Extract raw pixels at a reasonable working resolution to send to the worker.
    const { data, width, height } = getImageData(img, 256);

    const timeout = setTimeout(() => {
      mlWorkerReady = false;
      fallback();
    }, 3000);

    mlWorker.onmessage = (e) => {
      if (e.data && e.data.type === 'result') {
        clearTimeout(timeout);
        mlWorkerReady = true;
        resolve({ probs: e.data.probs, engine: e.data.engine || 'heuristic', elapsedMs: e.data.elapsedMs });
      }
    };

    mlWorker.postMessage({ type: 'infer', imageData: data, width, height });
  });
}

/* ============================================================
   MODEL ENGINE — TensorFlow.js with heuristic fallback
   ============================================================ */
async function initModel() {
  if (typeof tf === 'undefined') {
    modelStatus = 'heuristic';
    console.warn('TensorFlow.js not available — using heuristic engine.');
    return;
  }
  try {
    // Inspect model metadata first. A shipped placeholder model (zero
    // weights) must NOT drive diagnosis — the heuristic engine does.
    const manifest = await (await fetch(MODEL_URL)).json();
    if (manifest.placeholder) {
      modelStatus = 'heuristic';
      console.info('Placeholder model detected — using heuristic engine. Replace model.json + group1-shard1of1.bin and remove the placeholder flag to activate the neural engine.');
      return;
    }
    await tf.ready();
    model = await tf.loadLayersModel(MODEL_URL);
    modelStatus = 'neural';
    console.info('Neural model loaded.');
  } catch (err) {
    // Missing/corrupt model assets → deterministic fallback.
    modelStatus = 'heuristic';
    console.warn('Model load failed — switching to heuristic engine:', err.message);
  }
}

/**
 * Neural inference path — fully memory-safe via tf.tidy().
 * Pipeline: source image → center square crop → 224x224 → [-1,1] normalize
 *           → model.predict → softmax probability vector.
 */
function predictWithModel(img) {
  const elapsedMs = performance.now();
  const logits = tf.tidy(() => {
    const { data: pixels, width, height } = getImageData(img, INPUT_SIZE);
    const tensor = tf.browser
      .fromPixels({ data: pixels, width, height }, 3)
      .toFloat()
      .div(127.5)
      .sub(1.0)          // normalize [0,255] → [-1,1]
      .expandDims(0);    // [1, 224, 224, 3]
    return model.predict(tensor);
  });
  const probs = softmax(Array.from(logits.dataSync()));
  logits.dispose(); // release output tensor
  return { probs, elapsedMs: performance.now() - elapsedMs };
}

/**
 * Heuristic fallback classifier.
 * Extracts color-region + texture features and scores each of the 10
 * classes with deterministic evidence rules. Deliberately simple but
 * genuinely discriminative for the target leaf patterns.
 */
function analyzeHeuristic(img) {
  const t0 = performance.now();
  const { data, width, height } = getImageData(img, HEURISTIC_SIZE);

  let green = 0, yellow = 0, orange = 0, brown = 0, gray = 0, total = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    total++;
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    const sat = mx === 0 ? 0 : (mx - mn) / mx;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;

    if (g > 70 && g >= r && g >= b && g - b > 15 && g - r > 8) {
      green++;                                  // healthy green tissue
    } else if (r > 140 && g > 130 && b < 115 && r - g < 60 && g - b > 25) {
      yellow++;                                 // chlorotic yellowing / streak
    } else if (r > 150 && g > 70 && g < 170 && b < 90 && r - g > 40) {
      orange++;                                 // rust-orange pustules
    } else if (r > 90 && g > 45 && g < r && b < g && r - g > 20) {
      brown++;                                  // necrotic brown spots
    } else if (sat < 0.18 && lum > 90 && lum < 210) {
      gray++;                                   // pale gray/tan lesions
    }
  }

  // Texture measure: normalized luminance gradient (edge density).
  const edgeDensity = computeEdgeDensity(data, width, height);

  // Normalized feature fractions
  const fg = green / total;
  const fy = yellow / total;
  const fo = orange / total;
  const fb = brown / total;
  const fgray = gray / total;
  const fe = edgeDensity;

  // ----- Disease evidence rules (class 0..9) -----
  const rawScores = [
    // 0 Enset Bacterial Wilt: yellowing, green loss, little orange/gray
    fy * 3.0 + (1 - fg) * 1.5 + fb * 0.3 - fo * 2.2 - fgray * 0.5,
    // 1 Enset Streak Virus: streaky yellow on green + texture
    fy * 2.6 + fe * 1.6 + fg * 0.4 - fo * 1.6 - fb * 0.5,
    // 2 Coffee Leaf Rust: orange-yellow pustules dominate
    fo * 4.2 + fy * 1.8 - fg * 1.2 - fb * 0.6,
    // 3 Coffee Brown Eye Spot: brown spots w/ yellow halo, green intact
    fb * 3.2 + fy * 1.2 + fg * 0.3 - fo * 1.0 - fgray * 0.3,
    // 4 Northern Corn Leaf Blight: broad gray-tan lesions
    fgray * 3.4 + fg * 0.4 - fo * 1.2 - fy * 0.8 - fb * 0.3,
    // 5 Maize Gray Leaf Spot: narrow gray lesions, more green intact
    fgray * 2.6 + fg * 0.9 - fo * 0.8 - fy * 0.4 - fb * 0.2,
    // 6 Healthy: uniform green, minimal spots or texture
    fg * 3.6 + (1 - fy - fo - fb - fgray) * 1.5 - fe * 0.8,
    // 7 Coffee Berry Disease: dark brown/black lesions on berries & leaves
    fb * 3.0 + fo * 1.2 + fe * 0.6 - fg * 1.0 - fy * 0.4,
    // 8 Coffee Wilt Disease: sudden yellow wilting, low texture, brown wood
    fy * 2.2 + (1 - fg) * 1.4 + fb * 0.8 - fe * 1.2 - fo * 0.8,
    // 9 Maize Lethal Necrosis: yellow + necrotic brown + high texture
    fy * 2.4 + fb * 1.8 + fe * 1.4 - fg * 1.2 - fo * 0.6,
    // 10 Enset Mealybug: yellowing + green loss, low texture (sap-sucking)
    fy * 2.2 + (1 - fg) * 1.6 - fe * 1.4 - fo * 1.0 - fb * 0.3
  ];

  // Softmax with temperature 0.7 to sharpen decisive cases.
  const probs = softmax(rawScores, 0.7);
  return { probs, elapsedMs: performance.now() - t0 };
}

/* ---------------- Image Utilities ---------------- */

/** Center-square-crop an image and return its pixel data at `size`. */
function getImageData(img, size) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  const srcW = img.naturalWidth || img.width;
  const srcH = img.naturalHeight || img.height;
  const side = Math.min(srcW, srcH);
  const sx = (srcW - side) / 2;
  const sy = (srcH - side) / 2;

  ctx.drawImage(img, sx, sy, side, side, 0, 0, size, size);
  const imageData = ctx.getImageData(0, 0, size, size);

  // The canvas is discarded here; only raw pixel data is returned.
  return {
    data: imageData.data,
    width: size,
    height: size
  };
}

/** Simple Sobel-like edge density on luminance, normalized to [0,1]. */
function computeEdgeDensity(data, w, h) {
  let sum = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      const lr = 0.299 * data[i + 4] + 0.587 * data[i + 5] + 0.114 * data[i + 6];
      const ld = 0.299 * data[i + w * 4] + 0.587 * data[i + w * 4 + 1] + 0.114 * data[i + w * 4 + 2];
      sum += Math.abs(l - lr) + Math.abs(l - ld);
      count++;
    }
  }
  const avg = count > 0 ? sum / count : 0;
  return Math.min(1, avg / 110); // normalize gradient energy
}

/** Numerical softmax with optional temperature. */
function softmax(values, temperature = 1) {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp((v - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

function argMax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/* ============================================================
   FILE HANDLING — image lives in RAM only, discarded after scan
   ============================================================ */
function handleFileSelect(e) {
  const file = e.target.files && e.target.files[0];
  if (!file || !file.type.startsWith('image/')) return;

  // Stop the live camera stream before showing the preview
  stopCamera();

  // Clear any previous result
  hide(startScan);
  hide(cameraCard);
  hide(resultCard);
  hide(loadingCard);
  show(previewCard);

  // Release the previous image buffer
  clearRamImage();

  currentObjectUrl = URL.createObjectURL(file);

  const img = new Image();
  img.onload = () => {
    currentImage = img;
    previewImg.src = currentObjectUrl;
  };
  img.onerror = () => {
    clearRamImage();
    hide(previewCard);
    show(startScan);
  };
  img.src = currentObjectUrl;
}

/** Release the in-RAM image: revoke blob URL and drop all references. */
function clearRamImage() {
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
  currentImage = null;
  previewImg.removeAttribute('src');
}

/* ============================================================
   DIAGNOSIS PIPELINE
   ============================================================ */
async function runDiagnosis() {
  if (!currentImage) return;
  if (!catalog) {
    // Catalog failed to load — show a friendly error instead of crashing.
    hide(previewCard);
    hide(loadingCard);
    resultCard.className = 'result-card risk-inconclusive';
    resultIcon.textContent = '⚠️';
    resultTitle.textContent = I18N[lang].inconclusive;
    resultSubtitle.textContent = 'Catalog unavailable — check connection and retry.';
    visualMatch.textContent = 'The disease knowledge base could not be loaded. Please reload the page.';
    adviceList.innerHTML = '';
    confidenceValue.textContent = '—';
    confidenceFill.style.width = '0%';
    show(resultCard);
    return;
  }

  hide(previewCard);
  hide(resultCard);
  show(loadingCard);
  loadingText.textContent = I18N[lang].analyze;
  // Let the spinner paint before the (fast) synchronous analysis
  await new Promise((r) => setTimeout(r, 60));

  let probs;
  let engine = modelStatus === 'neural' && model ? 'neural' : 'heuristic';

  try {
    if (engine === 'neural') {
      probs = predictWithModel(currentImage).probs;
    } else {
      // Prefer the dedicated worker; fall back to main-thread heuristic.
      const r = await runInference(currentImage);
      probs = r.probs;
      engine = r.engine;
    }
  } catch (err) {
    console.warn('Inference failed — falling back to heuristic engine:', err.message);
    engine = 'heuristic';
    probs = analyzeHeuristic(currentImage).probs;
  }

  const topIdx = argMax(probs);
  const topConf = probs[topIdx];
  const confirmed = topConf >= CONFIDENCE_THRESHOLD;

  // Image processing complete → purge the photo from RAM.
  clearRamImage();

  const record = {
    timestamp: new Date().toISOString(),
    classId: confirmed ? topIdx : -1,
    className: confirmed ? catalog.classes[topIdx].name : 'Inconclusive Scan',
    crop: confirmed ? catalog.classes[topIdx].crop : '—',
    confidencePct: Math.round(topConf * 1000) / 10,
    riskLabel: confirmed ? catalog.classes[topIdx].risk.label : '⚪ Inconclusive',
    engine,
    inconclusive: !confirmed
  };

  logScan(record);
  updateAppStats(record.timestamp);

  if (confirmed) {
    lastResult = { cls: catalog.classes[topIdx], conf: topConf, engine, inconclusive: false };
    renderResult(catalog.classes[topIdx], topConf, engine);
  } else {
    lastResult = { conf: topConf, engine, inconclusive: true };
    renderInconclusive(topConf, engine);
  }

  // Allow the same file to be re-selected for a new scan
  photoInput.value = '';
}

/* ---------------- Rendering ---------------- */
function renderResult(cls, conf, engine) {
  const t = I18N[lang];
  resultCard.className = 'result-card risk-' + cls.risk.level;
  // Icon: use the leaf SVG icon (cls.icon may contain emoji from catalog)
  resultIcon.innerHTML = '<svg class="icon icon-xl"><use href="#i-leaf"/></svg>';
  // Haptic: disease detected + scan complete
  vibrate([15, 30]);
  // Dual taxonomy: scientific + Amharic name
  resultTitle.textContent =
    t.disease + (lang === 'am' ? cls.name_am : cls.name) +
    (lang === 'am' && cls.name !== cls.name_am ? ' (' + cls.name + ')' : '');
  resultSubtitle.textContent =
    (lang === 'am' ? cls.crop_am : cls.crop) + ' · ' +
    cls.pathogen + ' · ' + t.engine + ': ' + engine.toUpperCase();

  const pct = Math.round(conf * 100);
  const certainty = pct >= 85 ? t.high_certainty : t.confirmed;
  confidenceValue.textContent = pct + '% (' + certainty + ')';
  confidenceFill.style.width = Math.min(100, pct) + '%';

  visualMatch.textContent = loc(cls, 'visual_pattern', 'visual_pattern_am');

  adviceList.innerHTML = '';
  (lang === 'am' && cls.advice_am ? cls.advice_am : cls.advice).forEach((step) => {
    const li = document.createElement('li');
    li.textContent = step;
    adviceList.appendChild(li);
  });

  // Remove any previously-added dynamic blocks
  document.querySelectorAll('.dyn-block').forEach((n) => n.remove());

  // Dual Taxonomy: local Gurage names
  if (cls.local_names && cls.local_names.length) {
    appendResultBlock(resultCard, t.local_names, cls.local_names.join(' · '));
  }

  // Regional statistics
  if (cls.regional_stats) {
    appendResultBlock(
      resultCard,
      t.regional_title,
      lang === 'am' ? cls.regional_stats_am : cls.regional_stats,
      'regional'
    );
  }

  // Traditional practices
  if (cls.traditional_practices) {
    appendResultBlock(
      resultCard,
      t.traditional_title,
      lang === 'am' ? cls.traditional_practices_am : cls.traditional_practices,
      'traditional'
    );
  }

  // Landrace recommendation banner (enset classes)
  if (cls.landraces && cls.landraces.resistant) {
    const lr = lang === 'am' ? cls.landraces_am : cls.landraces;
    const html =
      '<strong>' + t.landrace_resistant + '</strong>' + lr.resistant.join(', ') +
      '<br /><strong>' + t.landrace_susceptible + '</strong>' + lr.susceptible.join(', ');
    appendResultBlock(resultCard, t.landrace_title, html, 'landrace');
  }

  // Resistant accessions (CBD)
  if (cls.resistant_accessions && cls.resistant_accessions.length) {
    appendResultBlock(
      resultCard,
      t.resistant_accessions,
      cls.resistant_accessions.join(', '),
      'accessions'
    );
  }

  // Tool sanitation priority (Enset BW)
  if (cls.sanitation) {
    appendResultBlock(
      resultCard,
      t.sanitation_title,
      lang === 'am' ? cls.sanitation.priority_am : cls.sanitation.priority,
      'sanitation'
    );
  }

  // Canopy advisor (coffee under enset shade)
  if (cls.canopy_advisor) {
    appendResultBlock(
      resultCard,
      t.canopy_title,
      lang === 'am' ? cls.canopy_advisor_am : cls.canopy_advisor,
      'canopy'
    );
  }

  // Seasonality
  if (cls.seasonality) {
    appendResultBlock(
      resultCard,
      t.seasonality_title,
      lang === 'am' ? cls.seasonality_am : cls.seasonality,
      'seasonality'
    );
  }

  // Transmission
  if (cls.transmission) {
    appendResultBlock(
      resultCard,
      t.transmission_title,
      lang === 'am' ? cls.transmission_am : cls.transmission,
      'transmission'
    );
  }

  // Spread rate badge
  if (cls.spread_rate) {
    const spreadLabel = {
      fast: t.spread_fast,
      moderate: t.spread_moderate,
      slow: t.spread_slow,
      none: t.spread_none
    }[cls.spread_rate] || cls.spread_rate;
    appendResultBlock(
      resultCard,
      t.spread_rate_title,
      '<span class="spread-badge spread-' + cls.spread_rate + '">' + spreadLabel + '</span>',
      'spread'
    );
  }

  // Yield impact
  if (cls.yield_impact) {
    appendResultBlock(
      resultCard,
      t.yield_impact_title,
      lang === 'am' ? cls.yield_impact_am : cls.yield_impact,
      'yield'
    );
  }

  // Monitoring frequency
  if (cls.monitoring_frequency) {
    appendResultBlock(
      resultCard,
      t.monitoring_title,
      lang === 'am' ? cls.monitoring_frequency_am : cls.monitoring_frequency,
      'monitoring'
    );
  }

  // Severity scale (1-5 stars)
  if (cls.severity_scale) {
    const stars = '★'.repeat(Math.min(5, cls.severity_scale)) +
      '☆'.repeat(Math.max(0, 5 - cls.severity_scale));
    appendResultBlock(
      resultCard,
      t.severity_title,
      '<span class="severity-stars">' + stars + '</span> (' + cls.severity_scale + '/5)',
      'severity'
    );
  }

  // Prevention tips
  if (cls.prevention_tips && cls.prevention_tips.length) {
    const tips = lang === 'am' && cls.prevention_tips_am ? cls.prevention_tips_am : cls.prevention_tips;
    const html = '<ul class="prevention-list">' + tips.map((tip) => '<li>' + tip + '</li>').join('') + '</ul>';
    appendResultBlock(resultCard, t.prevention_title, html, 'prevention');
  }

  // Traditional & easy home remedies (prominent)
  if (cls.home_remedies && cls.home_remedies.length) {
    const remedies = lang === 'am' && cls.home_remedies_am ? cls.home_remedies_am : cls.home_remedies;
    const html = '<ul class="remedy-list">' + remedies.map((r) => '<li>' + r + '</li>').join('') + '</ul>';
    appendResultBlock(resultCard, t.home_remedies_title, html, 'remedy');
  }

  // Food loss estimator (Enset BW)
  if (cls.food_loss) {
    appendFoodLossBlock(resultCard, cls, t);
  }

  hide(loadingCard);
  show(resultCard);
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/** Append a labelled info block to the result card. */
function appendResultBlock(card, title, contentHtml, type) {
  const block = document.createElement('div');
  block.className = 'dyn-block info-block' + (type ? ' info-' + type : '');
  const h3 = document.createElement('h3');
  h3.textContent = title;
  const p = document.createElement('p');
  p.innerHTML = contentHtml;
  block.appendChild(h3);
  block.appendChild(p);
  card.insertBefore(block, card.querySelector('.result-actions'));
}

/** Food Loss Estimator block with optional plant-count input (smooth, non-forcing). */
function appendFoodLossBlock(card, cls, t) {
  const fl = cls.food_loss;
  const block = document.createElement('div');
  block.className = 'dyn-block info-block info-foodloss';

  const h3 = document.createElement('h3');
  h3.textContent = t.food_loss_title;
  block.appendChild(h3);

  const note = document.createElement('p');
  note.textContent = lang === 'am' ? fl.note_am : fl.note;
  block.appendChild(note);

  const base = document.createElement('p');
  base.className = 'foodloss-base';
  base.textContent =
    '1 ' + t.food_loss_plants.replace(/\(s\)$/, '') +
    ' ≈ ' + fl.kocho_kg_per_plant + ' ' + t.food_loss_kocho +
    ' + ' + fl.bulla_kg_per_plant + ' ' + t.food_loss_bulla;
  block.appendChild(base);

  const estimator = document.createElement('div');
  estimator.className = 'foodloss-estimator';

  const label = document.createElement('label');
  label.htmlFor = 'foodLossCount';
  label.textContent = t.food_loss_plants + ': ';
  const input = document.createElement('input');
  input.type = 'number';
  input.id = 'foodLossCount';
  input.min = '1';
  input.value = '1';
  input.setAttribute('aria-label', t.food_loss_plants);

  const out = document.createElement('span');
  out.className = 'foodloss-out';

  const compute = () => {
    const n = Math.max(1, parseInt(input.value, 10) || 1);
    out.textContent =
      t.food_loss_total + ': ' +
      (n * fl.kocho_kg_per_plant) + ' ' + t.food_loss_kocho +
      ' + ' + (n * fl.bulla_kg_per_plant) + ' ' + t.food_loss_bulla;
  };
  input.addEventListener('input', compute);

  estimator.appendChild(label);
  estimator.appendChild(input);
  estimator.appendChild(out);
  block.appendChild(estimator);
  compute();

  card.insertBefore(block, card.querySelector('.result-actions'));
}

function renderInconclusive(conf, engine) {
  const t = I18N[lang];
  resultCard.className = 'result-card risk-inconclusive';
  resultIcon.innerHTML = '<svg class="icon icon-xl"><use href="#i-question"/></svg>';
  // Haptic: scan complete
  vibrate([15, 30]);
  resultTitle.textContent = t.inconclusive;
  resultSubtitle.textContent =
    t.inconclusive_desc + engine.toUpperCase();

  const pct = Math.round(conf * 100);
  confidenceValue.textContent = pct + '% (Below Safeguard)';
  confidenceFill.style.width = Math.min(100, pct) + '%';

  visualMatch.textContent =
    lang === 'am'
      ? 'ምስሉ በጣም የደበዘዘ፣ ብርሀኑ ደካማ ወይም የቅጠሉ ንድፍ አሻሚ ስለሆነ አስተማማኝ ምርመራ ማድረግ አይቻልም።'
      : 'The image is too blurry, poorly lit, or the leaf pattern is ambiguous for a reliable diagnosis.';

  const tips = lang === 'am'
    ? [
        'ቅጠሉን ወደ ደማቅ እና እኩል ብርሀን ያንቀሳቅሱት እና ካሜራውን በእርጋታ ይያዙ።',
        'የቅጠሉን ገጽ ብቻ በፍሬም ይሙሉ።',
        'ቅጠሉ ንጹህ እና ደረቅ መሆኑን ያረጋግጡ፣ ከዚያ እንደገና ፎቶ ያንሱ።'
      ]
    : [
        'Move the leaf into bright, even lighting and hold the camera steady.',
        'Fill more of the frame with a single leaf surface.',
        'Make sure the leaf is clean and dry, then retake the photo.'
      ];

  adviceList.innerHTML = '';
  tips.forEach((tip) => {
    const li = document.createElement('li');
    li.textContent = tip;
    adviceList.appendChild(li);
  });

  document.querySelectorAll('.dyn-block').forEach((n) => n.remove());

  hide(loadingCard);
  show(resultCard);
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ============================================================
   INDEXEDDB — text-only scan audit log
   ============================================================ */
function initDatabase() {
  return new Promise((resolve, reject) => {
    if (!('indexedDB' in window)) {
      console.warn('IndexedDB unavailable — history will not persist.');
      resolve();
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const d = e.target.result;
      if (!d.objectStoreNames.contains(DB_STORE)) {
        d.createObjectStore(DB_STORE, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = (e) => {
      db = e.target.result;
      resolve();
    };
    req.onerror = (e) => {
      console.warn('IndexedDB open failed:', e);
      resolve();
    };
  });
}

function logScan(record) {
  if (!db) return;
  const tx = db.transaction(DB_STORE, 'readwrite');
  tx.objectStore(DB_STORE).add(record);
}

function getAllScans() {
  return new Promise((resolve) => {
    if (!db) return resolve([]);
    const tx = db.transaction(DB_STORE, 'readonly');
    const req = tx.objectStore(DB_STORE).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => resolve([]);
  });
}

function clearAllScans() {
  return new Promise((resolve) => {
    if (!db) return resolve();
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

/** Load the national dashboard data (idempotent, lazy-load friendly). */
async function loadDashData() {
  if (dashData) return;
  try {
    const dres = await fetch('national-dashboard.json');
    dashData = await dres.json();
  } catch (err) {
    console.warn('Failed to load national dashboard:', err);
  }
}

/* ---------------- View Navigation ---------------- */
function switchView(view) {
  // Toggle nav active state
  document.querySelectorAll('.nav-item').forEach((item) => {
    item.classList.toggle('active', item.dataset.view === view);
  });

  // Update header title
  if (viewTitle) {
    const titleKey = 'view_' + view;
    viewTitle.textContent = I18N[lang][titleKey] || 'EnsetScan';
  }

  // Show/hide the three main views
  viewHome.hidden = view !== 'home';
  viewDashboard.hidden = view !== 'dashboard';
  viewHistory.hidden = view !== 'history';

  // When returning to Home, reset to the welcome screen if no scan is in progress
  if (view === 'home' && !currentImage && !lastResult) {
    hide(cameraCard);
    hide(previewCard);
    hide(loadingCard);
    hide(resultCard);
    show(startScan);
    stopCamera();
  }

  // Render content on demand; lazy-load dashboard data if not yet available
  if (view === 'dashboard') {
    if (dashData) {
      renderDashTab(dashTab);
    } else {
      loadDashData().then(() => {
        if (dashData) renderDashTab(dashTab);
      });
    }
  } else if (view === 'history') {
    openHistory();
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ---------------- National Dashboard ---------------- */
function openDashboard() {
  switchView('dashboard');
}

function switchDashTab(tab) {
  dashTab = tab;
  document.querySelectorAll('.dash-tab').forEach((t) => {
    t.classList.toggle('active', t.dataset.tab === tab);
  });
  if (dashData) {
    renderDashTab(tab);
  } else {
    loadDashData().then(() => {
      if (dashData) renderDashTab(tab);
    });
  }
}

/** Inline SVG illustration of a disease's visual signature (offline, zero-image). */
function dashSvg(kind) {
  const leaf = '<ellipse cx="50" cy="50" rx="40" ry="26" fill="#2e7d32" stroke="#1b5e20" stroke-width="2"/>';
  const vein = '<path d="M50 24 L50 76" stroke="#1b5e20" stroke-width="1.5" fill="none"/>';
  switch (kind) {
    case 'enset-wilt':
      return '<svg viewBox="0 0 100 100" width="90" height="90" aria-hidden="true">' + leaf +
        '<path d="M50 24 L50 76" stroke="#c62828" stroke-width="2" fill="none"/>' +
        '<circle cx="50" cy="40" r="6" fill="#f9a825"/><circle cx="50" cy="58" r="5" fill="#f9a825"/>' +
        '<path d="M30 30 Q50 20 70 30" stroke="#c62828" stroke-width="2" fill="none"/></svg>';
    case 'coffee-rust':
      return '<svg viewBox="0 0 100 100" width="90" height="90" aria-hidden="true">' + leaf + vein +
        '<circle cx="38" cy="42" r="4" fill="#ef6c00"/><circle cx="58" cy="50" r="4" fill="#ef6c00"/>' +
        '<circle cx="44" cy="62" r="3.5" fill="#ef6c00"/><circle cx="62" cy="36" r="3" fill="#ef6c00"/></svg>';
    case 'wheat-rust':
      return '<svg viewBox="0 0 100 100" width="90" height="90" aria-hidden="true">' + leaf + vein +
        '<path d="M30 40 L70 40 M30 52 L70 52 M30 64 L70 64" stroke="#f9a825" stroke-width="3" fill="none"/></svg>';
    case 'maize-mln':
      return '<svg viewBox="0 0 100 100" width="90" height="90" aria-hidden="true">' + leaf + vein +
        '<path d="M30 34 L70 34 M30 50 L70 50 M30 66 L70 66" stroke="#c62828" stroke-width="2.5" fill="none"/>' +
        '<path d="M30 42 L70 42 M30 58 L70 58" stroke="#f9a825" stroke-width="2" fill="none"/></svg>';
    case 'seedling-blight':
      return '<svg viewBox="0 0 100 100" width="90" height="90" aria-hidden="true">' + leaf + vein +
        '<circle cx="40" cy="40" r="5" fill="#8d6e63"/><circle cx="60" cy="55" r="5" fill="#8d6e63"/>' +
        '<circle cx="50" cy="70" r="4" fill="#8d6e63"/></svg>';
    default:
      return '<svg viewBox="0 0 100 100" width="90" height="90" aria-hidden="true">' + leaf + vein + '</svg>';
  }
}

function renderDashTab(tab) {
  if (!dashBody) return;
  if (!dashData) {
    dashBody.innerHTML = '<div class="dash-card"><p>' +
      (lang === 'am' ? 'ዳሽቦርድ ውሂብ አልተጫነም። እባክዎ ገጹን ያድሱ።' : 'Dashboard data failed to load. Please refresh the page.') +
      '</p></div>';
    return;
  }
  const t = I18N[lang];
  let html = '';

  if (tab === 'threats') {
    html = '<h3 class="dash-section-title">' + (lang === 'am' ? 'ብሔራዊ ስጋት አጠቃላይ እይታ' : 'National Threat Overview') + '</h3>';
    // Full detailed disease cards
    (dashData.diseases || []).forEach((d) => {
      const sym = lang === 'am' ? d.symptoms_am : d.symptoms;
      const prev = lang === 'am' ? d.prevention_am : d.prevention;
      const treat = lang === 'am' ? d.treatment_am : d.treatment;
      const local = lang === 'am' ? d.local_names_am : d.local_names;
      html += '<div class="dash-card dash-detail">' +
        '<div class="dash-card-head">' + dashSvg(d.svg) +
        '<div><h4>' + d.icon + ' ' + (lang === 'am' ? d.name_am : d.name) + '</h4>' +
        '<p class="dash-sub">' + (lang === 'am' ? d.crop_am : d.crop) + ' · ' + d.pathogen + '</p>' +
        '<p class="dash-risk">' + (lang === 'am' ? d.risk_am : d.risk) + ' ' + (lang === 'am' ? 'አደጋ' : 'Risk') + '</p></div></div>' +
        '<div class="dash-detail-grid">' +
        '<div class="dash-detail-col"><h5>🔍 ' + (lang === 'am' ? 'ምልክቶች' : 'Symptoms') + '</h5><ul>' + sym.map((s) => '<li>' + s + '</li>').join('') + '</ul></div>' +
        '<div class="dash-detail-col"><h5>🛡️ ' + (lang === 'am' ? 'መከላከያ' : 'Prevention') + '</h5><ul>' + prev.map((s) => '<li>' + s + '</li>').join('') + '</ul></div>' +
        '<div class="dash-detail-col"><h5>💊 ' + (lang === 'am' ? 'ህክምና' : 'Treatment') + '</h5><ul>' + treat.map((s) => '<li>' + s + '</li>').join('') + '</ul></div>' +
        '</div>' +
        '<p class="dash-impact"><strong>📈 ' + (lang === 'am' ? 'ተጽዕኖ፡ ' : 'Impact: ') + '</strong>' + (lang === 'am' ? d.impact_am : d.impact) + '</p>' +
        '<p class="dash-impact"><strong>🔄 ' + (lang === 'am' ? 'ስርጭት፡ ' : 'Transmission: ') + '</strong>' + (lang === 'am' ? d.transmission_am : d.transmission) + '</p>' +
        '<p class="dash-impact"><strong>📅 ' + (lang === 'am' ? 'ወቅት፡ ' : 'Season: ') + '</strong>' + (lang === 'am' ? d.season_am : d.season) + '</p>' +
        '<p class="dash-impact"><strong>🌐 ' + (lang === 'am' ? 'የአካባቢ ስሞች፡ ' : 'Local names: ') + '</strong>' + local.join(', ') + '</p>' +
        '<p class="dash-remedy"><strong>🏠 ' + (lang === 'am' ? 'ባህላዊ መፍትሄ፡ ' : 'Traditional remedy: ') + '</strong>' + (lang === 'am' ? d.traditional_remedy_am : d.traditional_remedy) + '</p>' +
        '</div>';
    });
  } else if (tab === 'remedies') {
    html = '<h3 class="dash-section-title">' + (lang === 'am' ? 'ባህላዊ መፍትሄ ማትሪክስ' : 'Indigenous Remedy Matrix') + '</h3>';
    dashData.remedies.forEach((r) => {
      html += '<div class="dash-card">' +
        '<div class="dash-card-head"><span class="dash-icon">' + r.icon + '</span>' +
        '<div><h4>' + (lang === 'am' ? r.name_am : r.name) + '</h4>' +
        '<p class="dash-sub"><strong>' + (lang === 'am' ? 'ዒላማ፡ ' : 'Targets: ') + '</strong>' + (lang === 'am' ? r.targets_am : r.targets) + '</p></div></div>' +
        '<p class="dash-impact">' + (lang === 'am' ? r.mechanism_am : r.mechanism) + '</p></div>';
    });
  } else if (tab === 'seasons') {
    html = '<h3 class="dash-section-title">' + (lang === 'am' ? 'ወቅታዊ ስጋት እና አካባቢዎች' : 'Seasonal Risk & Hotspots') + '</h3>';
    dashData.seasons.forEach((s) => {
      html += '<div class="dash-card' + (s.active ? ' dash-active' : '') + '">' +
        '<div class="dash-card-head"><span class="dash-icon">' + (s.active ? '🔴' : '⚪') + '</span>' +
        '<div><h4>' + (lang === 'am' ? s.name_am : s.name) + ' <small>(' + (lang === 'am' ? s.months_am : s.months) + ')</small></h4>' +
        '<p class="dash-sub">' + (lang === 'am' ? s.peak_am : s.peak) + '</p></div></div></div>';
    });
    html += '<h3 class="dash-section-title">' + (lang === 'am' ? 'አካባቢዎች' : 'Hotspots') + '</h3>';
    dashData.hotspots.forEach((h) => {
      html += '<div class="dash-card"><div class="dash-card-head"><span class="dash-icon">📍</span>' +
        '<div><h4>' + (lang === 'am' ? h.region_am : h.region) + '</h4>' +
        '<p class="dash-sub">' + (lang === 'am' ? h.threat_am : h.threat) + '</p></div></div></div>';
    });
  } else if (tab === 'varieties') {
    html = '<h3 class="dash-section-title">' + (lang === 'am' ? 'ተከላካይ ዝርያዎች' : 'Resistant Variety Directory') + '</h3>';
    dashData.varieties.forEach((v) => {
      const res = lang === 'am' ? v.resistant_am : v.resistant;
      const sus = lang === 'am' ? v.susceptible_am : v.susceptible;
      html += '<div class="dash-card"><div class="dash-card-head"><span class="dash-icon">🌱</span>' +
        '<div><h4>' + (lang === 'am' ? v.crop_am : v.crop) + '</h4>' +
        '<p class="dash-sub"><strong>' + (lang === 'am' ? 'ተከላካይ፡ ' : 'Resistant: ') + '</strong>' + res.join(', ') + '</p>' +
        (sus.length ? '<p class="dash-sub"><strong>' + (lang === 'am' ? 'ተጋላጭ፡ ' : 'Susceptible: ') + '</strong>' + sus.join(', ') + '</p>' : '') +
        '</div></div></div>';
    });
  }

  dashBody.innerHTML = html;
}

/* ---------------- History UI ---------------- */
async function openHistory() {
  const scans = await getAllScans();
  historyList.innerHTML = '';

  if (scans.length === 0) {
    historyEmpty.hidden = false;
  } else {
    historyEmpty.hidden = true;
    // Newest first
    scans
      .slice()
      .reverse()
      .forEach((s) => {
        const li = document.createElement('li');
        li.className = 'history-item';

        const icon = document.createElement('span');
        icon.className = 'h-icon';
        icon.innerHTML = s.inconclusive
          ? '<svg class="icon icon-lg"><use href="#i-question"/></svg>'
          : '<svg class="icon icon-lg"><use href="#i-leaf"/></svg>';
        li.appendChild(icon);

        const info = document.createElement('div');
        info.className = 'h-info';

        // Localize history entries by mapping stored classId back to the catalog
        let displayName = s.className;
        let displayCrop = s.crop;
        let displayRisk = s.riskLabel;
        if (!s.inconclusive && catalog && catalog.classes[s.classId]) {
          const cls = catalog.classes[s.classId];
          displayName = lang === 'am' ? cls.name_am : cls.name;
          displayCrop = lang === 'am' ? cls.crop_am : cls.crop;
          displayRisk = lang === 'am' ? cls.risk.label_am : cls.risk.label;
        }

        const title = document.createElement('div');
        title.className = 'h-title';
        title.textContent = displayName + (s.inconclusive ? '' : ' — ' + displayRisk);
        info.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'h-meta';
        meta.textContent =
          new Date(s.timestamp).toLocaleString() +
          (displayCrop !== '—' ? ' · ' + displayCrop : '') +
          ' · ' + s.engine;
        info.appendChild(meta);

        li.appendChild(info);

        const conf = document.createElement('span');
        conf.className = 'h-conf';
        conf.textContent = s.confidencePct.toFixed(1) + '%';
        li.appendChild(conf);

        historyList.appendChild(li);
      });
  }

}

function closeHistory() {
  // Clear the #history hash so it doesn't re-open on the next page load.
  if (window.location.hash === '#history') {
    history.replaceState(null, '', window.location.pathname + window.location.search);
  }
}

async function clearHistory() {
  await clearAllScans();
  closeHistory();
}

/* ---------------- UI Helpers ---------------- */
function show(section) { section.hidden = false; }
function hide(section) { section.hidden = true; }

function resetToUpload() {
  hide(cameraCard);
  hide(previewCard);
  hide(resultCard);
  hide(loadingCard);
  clearRamImage();
  photoInput.value = '';
  show(startScan);
  // Don't force-start the camera — let the user choose via the chooser.
  stopCamera();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
