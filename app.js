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
const CATALOG_TIMEOUT_MS = 10000;   // 10s fetch timeout (bugs #5, #17)
const INPUT_SIZE = 224;             // neural input 224x224
const HEURISTIC_SIZE = 96;          // heuristic working resolution
const DB_NAME = 'ensetscan-db';
const DB_VERSION = 1;
const DB_STORE = 'scans';

/* ---------------- Built-in minimal catalog (offline fallback, bugs #5/#17) ------
   Used when the full disease-catalog.json cannot be fetched within 10s.
   Contains all 17 classes with complete EN/AM content so the offline dashboard
   never shows placeholder text for Prevention/Seasonality/Remedies/Varieties. */
const MINIMAL_CATALOG = {
  version: "offline-minimal",
  threshold: { min_confidence: 0.65, label: "65% Confidence Safeguard" },
  classes: [
    { class_id: 0, crop: "Enset", crop_am: "እንሰት", name: "Enset Bacterial Wilt", name_am: "የእንሰት ባክቴሪያ ዊልት", pathogen: "Xanthomonas vasicola pv. musacearum", icon: "⚠️", risk: { level: "critical", label: "🔴 Critical Risk", label_am: "🔴 ከፍተኛ አደጋ", color: "#c62828" },
      visual_pattern: "Central heart-leaf yellowing, petiole drooping, and yellow bacterial fluid evident when the pseudostem is cut.", visual_pattern_am: "የመሃል ቅጠል መገርጠት፣ የግንድ መደፈር እና ግንዱ ሲቆረጥ ቢጫ የባክቴሪያ ፈሳሽ መታየት።", advice: ["Uproot and bury the infected plant immediately in a deep pit away from the farm.", "Flame-sterilize or wash slicing knives with a 10% bleach solution after every cut.", "Do not replant Enset in the same hole for 6–12 months; keep the soil exposed to direct sunlight."], advice_am: ["የተያዘውን ተክል ወዲያውኑ ቆፍረው ከእርሻ ርቆ በጥልቅ ጉድጓድ ውስጥ ይቅበሩ።", "ከእያንዳንዱ ቁረጥ በኋላ ቢላዎችን በእሳት ወይም በ10% ብሊች ይታጠቡ።", "ለ6–12 ወራት በተመሳሳይ ጉድጓድ እንሰት አይተክሉ፤ አፈሩን ለፀሀይ ያጋልጡ።"], prevention_tips: ["Plant early and space mats widely so tools never touch neighbouring plants.", "Bury infected debris deep and keep the pit open to sunlight."], prevention_tips_am: ["ቀድመው ይትከሉ እና እንሰቶችን በስፋት ያርቁ።", "የተያዘውን ቅሪት በጥልቅ ይቅበሩ እና ጉድጓዱን ለፀሀይ ያጋልጡ።"], seasonality: "Rainy season (June–September), when soil moisture is high.", seasonality_am: "የዝናብ ወቅት (ሰኔ–መስከረም)፣ የአፈር እርጥበት ከፍ ባለበት ጊዜ።", home_remedies: ["Traditional: sprinkle ash from burnt enset leaves around the base of healthy plants to deter spread.", "Easy: never re-use a cutting knife without passing it through fire first — this is free and stops the disease moving plant to plant."], home_remedies_am: ["ባህላዊ፡ ጤናማ ተክሎችን ለመከላከል በእንሰት ስር ከተቃጠለ ቅጠል የተገኘ አመድ ይረጩ።", "ቀላል፡ የመቁረጫ ቢላዋን በእሳት ሳያሳልፉ እንደገና አይጠቀሙ — ይህ ነጻ ነው እና በሽታውን ከተክል ወደ ተክል እንዳይሸጋገር ይከላከላል።"], landraces: { resistant: ["Lemat", "Beshute", "Yiregiye"], susceptible: ["Ageremremat", "Shertye", "Kibnar"] }, landraces_am: { resistant: ["ለማት", "በሹቴ", "ይረጊዬ"], susceptible: ["አገረምረማት", "ሸርጤ", "ክብናር"] },
       },
    { class_id: 1, crop: "Enset", crop_am: "እንሰት", name: "Enset Streak Virus", name_am: "የእንሰት ስትሪክ ቫይረስ", pathogen: "Enset streak virus (BSV)", icon: "🟡", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Continuous yellow mosaic lines and broken streaking running parallel along the leaf veins.", visual_pattern_am: "በቅጠል ስሮች ላይ ትይዩ የሆኑ ቀጣይ ቢጫ ሞዛይክ መስመሮች እና የተሰበሩ ጅራቶች።", advice: ["Prune and burn individual heavily streaked leaves to limit viral spread.", "Spray a mild soap-water or neem oil solution on leaf undersides to kill mealybugs and aphids.", "Avoid transferring planting materials from infected mats to new fields."], advice_am: ["በጣም የተጠቁትን ቅጠሎች ቆርጠው ያቃጥሉ።", "ሜሊባግ እና አፊድን ለመግደል በቅጠል ስር ሳሙና-ውሃ ወይም የኒም ዘይት ይረጩ።", "ከተያዙ እንሰቶች ወደ አዲስ እርሻ የመትከያ ቁሳቁስ አያስተላልፉ።"], prevention_tips: ["Only take suckers from healthy-looking mother plants.", "Keep the garden weed-free to reduce insect hiding places."], prevention_tips_am: ["ችግኞችን የሚወስዱት ከጤናማ እናት ተክሎች ብቻ ይሁን።", "ነፍሳት የሚደበቁበትን አረም ለማስወገድ አትክልቱን ንጹህ ያድርጉ።"], seasonality: "Year-round; worst during dry spells when insects are active.", seasonality_am: "ዓመቱን ሙሉ፤ ነፍሳት በሚንቀሳቀሱበት ደረቅ ወቅት የባሰ ነው።", home_remedies: ["Traditional: hang neem leaves in the garden — their smell repels aphids and mealybugs.", "Easy: mix a spoon of dish soap in 2L water and spray the underside of leaves once a week."], home_remedies_am: ["ባህላዊ፡ በአትክልቱ ውስጥ የኒም ቅጠሎችን ይስቀሉ — ሽታቸው አፊድን እና ሜሊባግን ያባርራል።", "ቀላል፡ አንድ ማንኪያ ሳሙና በ2 ሊትር ውሃ ቀላቅለው በሳምንት አንድ ጊዜ ቅጠሎችን ስር ይረጩ።"], landraces: { resistant: ["Lemat", "Beshute"], susceptible: ["Ageremremat", "Kibnar"] }, landraces_am: { resistant: ["ለማት", "በሹቴ"], susceptible: ["አገረምረማት", "ክብናር"] },
       },
    { class_id: 2, crop: "Coffee", crop_am: "ቡና", name: "Coffee Leaf Rust", name_am: "የቡና ቅጠል ዝገት", pathogen: "Hemileia vastatrix", icon: "🟠", risk: { level: "high", label: "🟠 High Risk", label_am: "🟠 ከፍተኛ አደጋ", color: "#ef6c00" },
      visual_pattern: "Bright orange or yellow powdery spots on the underside of coffee leaves.", visual_pattern_am: "በቡና ቅጠል ስር ደማቅ ብርቱካናማ ወይም ቢጫ የዱቄት ነጠብጣቦች።", advice: ["Prune overlapping canopy shade branches to increase airflow and sunlight exposure.", "Apply an organic copper-based fungicide spray prior to the main rainy season.", "Remove heavily infected lower foliage lying close to the soil surface."], advice_am: ["የአየር ዝውውር እና የፀሀይ ብርሀን ለማሳደግ የተደራረቡ የጥላ ቅርንጫፎችን ይቁረጡ።", "ከዋናው የዝናብ ወቅት በፊት ኦርጋኒክ የመዳብ ፈንገስ መርጫ ይረጩ።", "ከአፈር ገጽ አጠገብ ያሉ በጣም የተያዙ የታችኛው ቅጠሎችን ያስወግዱ።"], prevention_tips: ["Keep coffee rows airy by spacing trees well.", "Harvest and remove fallen leaves from under the trees."], prevention_tips_am: ["የቡና ዛፎችን በስፋት በማስቀመጥ አየር እንዲያገኙ ያድርጉ።", "ከዛፎች ስር የወደቁ ቅጠሎችን ይሰብስቡ እና ያስወግዱ።"], seasonality: "Late rainy season to early dry season (August–November).", seasonality_am: "ከዝናብ መጨረሻ እስከ ደረቅ ወቅት መጀመሪያ (ነሀሴ–ህዳር)።", home_remedies: ["Traditional: plant a row of maize or banana as a living windbreak around the coffee field.", "Easy: pick off and burn the very first rusted leaves — one early action can stop an outbreak."], home_remedies_am: ["ባህላዊ፡ በቡና እርሻ ዙሪያ የበቆሎ ወይም የሙዝ ረድፍ እንደ የነፋስ መከላከያ ይትከሉ።", "ቀላል፡ የመጀመሪያዎቹን የተያዙ ቅጠሎች ቀልጠው ያቃጥሉ — ቀደም ያለ እርምጃ ወረርሽኝን ይከላከላል።"], landraces: { resistant: ["Gu-18", "Gu-1", "Gu-4"], susceptible: ["Geisha", "Caturra"] }, landraces_am: { resistant: ["ጉ-18", "ጉ-1", "ጉ-4"], susceptible: ["ጌሻ", "ካቱራ"] },
       },
    { class_id: 3, crop: "Coffee", crop_am: "ቡና", name: "Coffee Brown Eye Spot", name_am: "የቡና ቡናማ አይን ነጠብጣብ", pathogen: "Cercospora coffeicola", icon: "🟡", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Small circular brown spots with grayish-white centers surrounded by a bright yellow halo.", visual_pattern_am: "በደማቅ ቢጫ ቀለበት የተከበቡ ግራጫ-ነጭ መሃል ያላቸው ትናንሽ ክብ ቡናማ ነጠብጣቦች።", advice: ["Apply compost or potassium-rich organic fertilizer to strengthen nutrient-deficient plants.", "Place dry grass mulch around the root base to keep soil moisture and temperature stable.", "Provide balanced shade in nursery beds to prevent heat stress."], advice_am: ["የንጥረ-ምግብ እጥረት ያለባቸውን ተክሎች ለማጠናከር ማዳበሪያ ወይም ፖታስየም የበለጸገ ኦርጋኒክ ማዳበሪያ ይጠቀሙ።", "የአፈር እርጥበት እና ሙቀት እንዲረጋ በስር ደረቅ ሳር ይከልሉ።", "የሙቀት ጫናን ለመከላከል በችግኝ ቦታዎች ሚዛናዊ ጥላ ይስጡ።"], prevention_tips: ["Use only healthy seeds and keep nursery soil clean.", "Mulch the root base to keep soil moisture and temperature stable."], prevention_tips_am: ["ጤናማ ዘሮችን ብቻ ይጠቀሙ እና የችግኝ አፈርን ንጹህ ያድርጉ።", "የአፈር እርጥበት እና ሙቀት እንዲረጋ በስር ይከልሉ።"], seasonality: "Dry season and sun-exposed nursery beds.", seasonality_am: "ደረቅ ወቅት እና ለፀሀይ የተጋለጡ የችግኝ ቦታዎች።", home_remedies: ["Traditional: water seedlings with diluted compost tea once a month to strengthen young plants.", "Easy: raise nursery beds and thin seedlings so air moves freely between them."], home_remedies_am: ["ባህላዊ፡ ችግኞችን ለማጠናከር በየወሩ በተቀላቀለ የማዳበሪያ ሻይ ያጠጡ።", "ቀላል፡ የችግኝ ቦታዎችን ከፍ ያድርጉ እና ችግኞችን አየር እንዲያገኙ ያርቁ።"], landraces: { resistant: ["No known resistant varieties"], susceptible: ["Catuai", "Typica"] }, landraces_am: { resistant: ["የታወቀ ተከላካይ ዝርያ የለም"], susceptible: ["ካቱዋይ", "ቲፒካ"] },
       },
    { class_id: 4, crop: "Maize", crop_am: "በቆሎ", name: "Northern Corn Leaf Blight", name_am: "የሰሜን በቆሎ ቅጠል በሽታ", pathogen: "Exserohilum turcicum", icon: "🟠", risk: { level: "high", label: "🟠 High Risk", label_am: "🟠 ከፍተኛ አደጋ", color: "#ef6c00" },
      visual_pattern: "Large, long, cigar-shaped grayish-tan lesions running parallel along the leaf blade.", visual_pattern_am: "በቅጠሉ ላይ ትይዩ የሆኑ ትላልቅ ሲጋራ ቅርጽ ያላቸው ግራጫ-ቡናማ ቁስሎች።", advice: ["Rotate crops next season with leguminous, nitrogen-fixing crops (haricot beans or peas).", "Clear and burn or compost leftover maize stalks and leaf residue after harvest.", "Plant certified disease-resistant hybrid seed varieties recommended by local extension agents."], advice_am: ["በሚቀጥለው ወቅት ከጥራጥሬ ሰብሎች (ባቄላ ወይም አተር) ጋር ይቀያይሩ።", "ከመኸር በኋላ የቀሩ የበቆሎ ግንዶችን እና ቅጠሎችን ያጽዱ እና ያቃጥሉ።", "በአካባቢ ባለሙያዎች የሚመከሩ የተረጋገጡ ተከላካይ የድብል ዘሮችን ይትከሉ።"], prevention_tips: ["Burn or compost all maize stalks after harvest.", "Plant resistant hybrid seeds recommended locally."], prevention_tips_am: ["ከመኸር በኋላ ሁሉንም የበቆሎ ግንዶች ያቃጥሉ ወይም ያበስብሉ።", "በአካባቢ የሚመከሩ ተከላካይ ድብል ዘሮችን ይትከሉ።"], seasonality: "Cool, humid periods (July–September).", seasonality_am: "ቀዝቃዛና እርጥበታማ ጊዜያት (ሀምሌ–መስከረም)።", home_remedies: ["Traditional: interplant maize with beans — the beans shade the soil and reduce leaf wetness.", "Easy: never plant maize on the same plot two seasons in a row; the rotation is free and effective."], home_remedies_am: ["ባህላዊ፡ በቆሎ ከባቄላ ጋር ይቀላቅሉ — ባቄላ አፈርን ይከልላል እና የቅጠል እርጥበትን ይቀንሳል።", "ቀላል፡ በተመሳሳይ መሬት ሁለት ወቅት በተከታታይ በቆሎ አይትከሉ፤ ሽክርክሪቱ ነጻ እና ውጤታማ ነው።"], landraces: { resistant: ["BH-660", "BH-540"], susceptible: ["Local unimproved"] }, landraces_am: { resistant: ["ቢ-ኤች-660", "ቢ-ኤች-540"], susceptible: ["አካባቢው ያልተሻሻለ"] },
       },
    { class_id: 5, crop: "Maize", crop_am: "በቆሎ", name: "Maize Gray Leaf Spot", name_am: "የበቆሎ ግራጫ ቅጠል ነጠብጣብ", pathogen: "Cercospora zeae-maydis", icon: "🟡", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Strict rectangular, narrow tan/gray lesions bounded tightly by parallel leaf veins.", visual_pattern_am: "በትይዩ ቅጠል ስሮች የታሰሩ ጥብቅ አራት ማዕዘን ቅርጽ ያላቸው ጠባብ ቡናማ/ግራጫ ቁስሎች።", advice: ["Clear post-harvest crop residue completely to eliminate overwintering fungal spores.", "Maintain proper row spacing during planting to avoid trapping excess humidity between plants.", "Avoid continuous maize monocropping in the same field plot."], advice_am: ["የክረምት ፈንገስ ስፖሮችን ለማስወገድ ከመኸር በኋላ ቅሪትን ሙሉ በሙሉ ያጽዱ።", "በተክሎች መካከል እርጥበት እንዳይታሰር ትክክለኛ የረድፍ ክፍተት ይጠብቁ።", "በተመሳሳይ እርሻ ተከታታይ የበቆሎ ሞኖክሮፕ ያስወግዱ።"], prevention_tips: ["Clear all post-harvest residue completely.", "Space rows well to avoid trapping humidity."], prevention_tips_am: ["ከመኸር በኋላ ቅሪትን ሙሉ በሙሉ ያጽዱ።", "እርጥበት እንዳይታሰር ረድፎችን በስፋት ያስቀምጡ።"], seasonality: "Humid, warm months after the canopy closes.", seasonality_am: "ጥላው ከተዘጋ በኋላ እርጥበታማ እና ሞቃት ወራት።", home_remedies: ["Traditional: leave the plot fallow one season and plant haricot beans instead.", "Easy: thin crowded plants so sunlight reaches every leaf."], home_remedies_am: ["ባህላዊ፡ ለአንድ ወቅት መሬቱን አርፍ ያድርጉ እና በምትኩ ባቄላ ይትከሉ።", "ቀላል፡ የተጨናነቁ ተክሎችን በማቅለል ፀሀይ እያንዳንዱን ቅጠል እንዲደርስ ያድርጉ።"], landraces: { resistant: ["Resistant hybrids (check local extension)"], susceptible: ["Local open-pollinated"] }, landraces_am: { resistant: ["ተከላካይ ድብልዎች (የአካባቢ ባለሙያን ይጠይቁ)"], susceptible: ["የአካባቢ ክፍት-የአበባ ዝርያ"] },
       },
    { class_id: 6, crop: "All Crops", crop_am: "ሁሉም ሰብሎች", name: "Healthy Leaf Tissue", name_am: "ጤናማ ቅጠል", pathogen: "None — Baseline Control", icon: "🟢", risk: { level: "normal", label: "🟢 Normal", label_am: "🟢 መደበኛ", color: "#2e7d32" },
      visual_pattern: "Uniform green pigmentation across Enset, Coffee, or Maize leaves without spots or streaks.", visual_pattern_am: "ያለ ነጠብጣብ ወይም ጅራት በእንሰት፣ ቡና ወይም በቆሎ ቅጠሎች ላይ ወጥ የሆነ አረንጓዴ ቀለም።", advice: ["Maintain standard weeding, mulching, and organic composting routines.", "Inspect fields weekly for early symptoms during high-humidity periods.", "Continue good field sanitation to keep crops vigorous and productive."], advice_am: ["መደበኛ አረም ማጽዳት፣ መከለል እና ኦርጋኒክ ማዳበሪያ ልምዶችን ይጠብቁ።", "በከፍተኛ እርጥበት ወቅት ሳምንታዊ የእርሻ ምርመራ ያድርጉ።", "ሰብሎችን ጠንካራ ለማድረግ ጥሩ የእርሻ ንጽህና ይቀጥሉ።"], prevention_tips: ["Keep tools clean between fields.", "Compost plant residue away from the growing area."], prevention_tips_am: ["መሳሪያዎችን በእርሻዎች መካከል ንጹህ ያድርጉ።", "የተክል ቅሪቶችን ከእርሻው ርቀው ያበስብሉ።"], seasonality: "All year — no disease window.", seasonality_am: "ዓመቱን ሙሉ — ምንም የበሽታ ወቅት የለም።", home_remedies: ["Traditional: keep the farm clean with regular weeding and compost.", "Easy: walk the field weekly; early signs are easier to manage than late ones."], home_remedies_am: ["ባህላዊ፡ እርሻውን በመደበኛ አረም ማጽዳት እና ማዳበሪያ ንጹህ ያድርጉ።", "ቀላል፡ በሳምንት አንድ ጊዜ እርሻውን ይመርምሩ፤ ቀደም ያለ ምልክት በቀላሉ ይታከማል።"], landraces: { resistant: [], susceptible: [] }, landraces_am: { resistant: [], susceptible: [] },
       },
    { class_id: 7, crop: "Coffee", crop_am: "ቡና", name: "Coffee Berry Disease", name_am: "የቡና ፍሬ በሽታ", pathogen: "Colletotrichum kahawae", icon: "🟠", risk: { level: "high", label: "🟠 High Risk", label_am: "🟠 ከፍተኛ አደጋ", color: "#ef6c00" },
      visual_pattern: "Dark sunken lesions on green berries that turn black and rot; leaf spots may also appear.", visual_pattern_am: "በአረንጓዴ ፍሬዎች ላይ ጠቆር ያሉ የተጠመቁ ቁስሎች ወደ ጥቁር እና መበስበስ የሚሄዱ፤ የቅጠል ነጠብጣቦችም ሊታዩ ይችላሉ።", advice: ["Plant resistant accessions such as Gu-18, Gu-1, and Gu-4 where available.", "Remove and destroy infected berries early to reduce inoculum.", "Improve air circulation and avoid dense, humid canopies."], advice_am: ["በሚገኝበት ቦታ እንደ Gu-18፣ Gu-1 እና Gu-4 ያሉ ተከላካይ ዝርያዎችን ይትከሉ።", "የበሽታ ስርጭትን ለመቀነስ የተያዙ ፍሬዎችን ቀደም ብለው ያስወግዱ እና ያጥፉ።", "የአየር ዝውውርን ያሻሽሉ እና ጥቅጥቅ ያለ እርጥበታማ ጥላን ያስወግዱ።"], prevention_tips: ["Prune for good airflow and keep branches off the ground.", "Use resistant accessions Gu-18, Gu-1, or Gu-4."], prevention_tips_am: ["ለጥሩ የአየር ዝውውር ይቁረጡ እና ቅርንጫፎችን ከመሬት ያርቁ።", "ተከላካይ ዝርያዎችን Gu-18፣ Gu-1 ወይም Gu-4 ይጠቀሙ።"], seasonality: "Main rainy season (May–August), when berries are still green.", seasonality_am: "ዋና የዝናብ ወቅት (ግንቦት–ነሀሴ)፣ ፍሬዎቹ አረንጓዴ ሲሆኑ።", home_remedies: ["Traditional: strip and burn all mummified berries during pruning — they carry next season's spores.", "Easy: pick berries early and often; hand removal is the cheapest control."], home_remedies_am: ["ባህላዊ፡ በመቁረጥ ወቅት የደረቁ ፍሬዎችን ሁሉ ቀልቀው ያቃጥሉ።", "ቀላል፡ ፍሬዎችን ቀድመው እና በተደጋጋሚ ይልቀሙ፤ በእጅ ማስወገድ በጣም ርካሹ መቆጣጠሪያ ነው።"], landraces: { resistant: ["Gu-18", "Gu-1", "Gu-4"], susceptible: ["Geisha"] }, landraces_am: { resistant: ["ጉ-18", "ጉ-1", "ጉ-4"], susceptible: ["ጌሻ"] },
       },
    { class_id: 8, crop: "Coffee", crop_am: "ቡና", name: "Coffee Wilt Disease", name_am: "የቡና ዊልት በሽታ", pathogen: "Gibberella xylarioides", icon: "🔴", risk: { level: "critical", label: "🔴 Critical Risk", label_am: "🔴 ከፍተኛ አደጋ", color: "#c62828" },
      visual_pattern: "Sudden wilting and yellowing of branches; dark blue-black streaks in the wood; plant death.", visual_pattern_am: "ድንገተኛ መደረቅ እና የቅርንጫፎች መገርጠት፤ በእንጨቱ ውስጥ ጥቁር ሰማያዊ ጅራቶች፤ የተክሉ ሞት።", advice: ["Uproot and burn infected trees immediately to prevent spread.", "Sterilize pruning tools between trees with flame or bleach.", "Use certified disease-free seedlings for replanting."], advice_am: ["ስርጭትን ለመከላከል የተያዙ ዛፎችን ወዲያውኑ ቆፍረው ያቃጥሉ።", "በዛፎች መካከል የመቁረጫ መሳሪያዎችን በእሳት ወይም ብሊች ያጸዱ።", "ለመትከል የተረጋገጡ ከበሽታ ነጻ ችግኞችን ይጠቀሙ።"], prevention_tips: ["Use only certified disease-free seedlings.", "Avoid wounding trunks during weeding or shade work."], prevention_tips_am: ["የተረጋገጡ ከበሽታ ነጻ ችግኞችን ብቻ ይጠቀሙ።", "በአረም ማጽዳት ወይም በጥላ ስራ ወቅት ግንዶችን ከመቁሰል ይቆጠቡ።"], seasonality: "Year-round; peaks after heavy rain or drought stress.", seasonality_am: "ዓመቱን ሙሉ፤ ከከባድ ዝናብ ወይም ድርቅ ጫና በኋላ ይባሳል።", home_remedies: ["Traditional: burn wilted stumps and leave the hole open to sunlight for the whole dry season.", "Easy: dip your machete in fire or bleach between every single tree — it costs nothing but discipline."], home_remedies_am: ["ባህላዊ፡ የደረቁ ጉቶዎችን ያቃጥሉ እና ጉድጓዱን ለደረቅ ወቅት ሙሉ ለፀሀይ ይክፈቱ።", "ቀላል፡ በእያንዳንዱ ዛፍ መካከል መቁረጫዎን በእሳት ወይም ብሊች ይንከሩ — ነጻ ነው ነገር ግን ቅድሚያ ይጠይቃል።"], landraces: { resistant: ["No known resistant varieties"], susceptible: ["Geisha", "F-59"] }, landraces_am: { resistant: ["የታወቀ ተከላካይ ዝርያ የለም"], susceptible: ["ጌሻ", "ኤፍ-59"] },
       },
    { class_id: 9, crop: "Maize", crop_am: "በቆሎ", name: "Maize Lethal Necrosis", name_am: "የበቆሎ ገዳይ ኔክሮሲስ", pathogen: "Maize chlorotic mottle virus (MCMV) complex", icon: "🔴", risk: { level: "critical", label: "🔴 Critical Risk", label_am: "🔴 ከፍተኛ አደጋ", color: "#c62828" },
      visual_pattern: "Yellowing and necrosis of leaves with severe stunting; dead heart; plant death.", visual_pattern_am: "የቅጠሎች መገርጠት እና ኔክሮሲስ ከከባድ ድውየት ጋር፤ የልብ መሞት፤ የተክል ሞት።", advice: ["Rogue and destroy infected plants immediately to reduce virus spread.", "Control insect vectors (thrips, aphids, beetles) that transmit the virus.", "Rotate with non-cereal crops and use certified seed."], advice_am: ["የቫይረስ ስርጭትን ለመቀነስ የተያዙ ተክሎችን ወዲያውኑ ቆፍረው ያጥፉ።", "ቫይረሱን የሚያስተላልፉ ነፍሳትን (ትሪፕስ፣ አፊድ፣ ጥንዚዛ) ይቆጣጠሩ።", "ከእህል ሰብሎች ጋር ይቀያይሩ እና የተረጋገጠ ዘር ይጠቀሙ።"], prevention_tips: ["Rotate with beans and use certified seed.", "Control weeds that host the insect vectors."], prevention_tips_am: ["ከባቄላ ጋር ይቀያይሩ እና የተረጋገጠ ዘር ይጠቀሙ።", "ነፍሳትን የሚያስተናግዱ አረሞችን ይቆጣጠሩ።"], seasonality: "Warm season; spreads rapidly with insect vectors.", seasonality_am: "ሞቃት ወቅት፤ ከነፍሳት ተሸካሚዎች ጋር በፍጥነት ይተላለፋል።", home_remedies: ["Traditional: plant maize early so the crop matures before insect pressure peaks.", "Easy: pull out and destroy any sick plant the moment you see it — one plant saved is the whole field."], home_remedies_am: ["ባህላዊ፡ ሰብሉ ከነፍሳት ጫና በፊት እንዲበስል በቆሎን ቀድመው ይትከሉ።", "ቀላል፡ የታመመ ተክል ሲያዩ ወዲያውኑ ቆፍረው ያጥፉ — አንድ የተከለለ ተክል ሙሉ እርሻን ያድናል።"], landraces: { resistant: ["Resistant hybrids (check local extension)"], susceptible: ["Susceptible local maize"] }, landraces_am: { resistant: ["ተከላካይ ድብልዎች (የአካባቢ ባለሙያን ይጠይቁ)"], susceptible: ["የአካባቢ ስሱ በቆሎ"] },
       },
    { class_id: 10, crop: "Enset", crop_am: "እንሰት", name: "Enset Mealybug", name_am: "የእንሰት ሜሊባግ", pathogen: "Cataenococcus ensete", icon: "🐛", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Leaf yellowing and stunting; white waxy mealybug clusters at the corm base and roots.", visual_pattern_am: "የቅጠል መገርጠት እና ድውየት፤ በግንዱ ስር እና በስሮች ላይ ነጭ ሰም ያላቸው የሜሊባግ ስብስቦች።", advice: ["Uncover the upper corm/root area and apply a generous layer of fresh, dry wood ash directly around the base.", "Remove and destroy heavily infested plants to stop spread to healthy mats.", "Avoid moving soil or planting material from infested gardens to clean fields."], advice_am: ["የላይኛውን ግንድ/ስር አካባቢ ከፍተው ትኩስ ደረቅ የእንጨት አመድ በቀጥታ በስሩ ይረጩ።", "ወደ ጤናማ እንሰቶች እንዳይስፋፋ በጣም የተያዙ ተክሎችን አስወግደው ያጥፉ።", "ከተያዙ አትክልቶች ወደ ንጹህ እርሻ አፈር ወይም የመትከያ ቁሳቁስ ከማንቀሳቀስ ይቆጠቡ።"], prevention_tips: ["Only take suckers from clean, healthy mother plants.", "Keep the base free of weeds and debris where ants can nest."], prevention_tips_am: ["ችግኞችን ከንጹህ ጤናማ እናት ተክሎች ብቻ ይውሰዱ።", "ጉንዳኖች የሚኖሩበትን አረም እና ቅሪት ከስሩ ያጽዱ።"], seasonality: "Year-round; worse in dry spells when plants are stressed.", seasonality_am: "ዓመቱን ሙሉ፤ ተክሎች በሚጨነቁበት ደረቅ ወቅት የባሰ ነው።", home_remedies: ["Traditional: dust fine wood ash over the corm base — it dries out and kills the soft-bodied mealybugs.", "Easy: spray a garlic + hot pepper + soap water mix on the base to repel young mealybugs."], home_remedies_am: ["ባህላዊ፡ በግንዱ ስር የእንጨት አመድ ይረጩ — ሜሊባግን ያደርቃል እና ይገድላል።", "ቀላል፡ የነጭ ሽንኩርት + በርበሬ + ሳሙና ውሃ በስሩ ይረጩ።"], landraces: { resistant: ["No known resistant varieties"], susceptible: ["Local susceptible"] }, landraces_am: { resistant: ["የታወቀ ተከላካይ ዝርያ የለም"], susceptible: ["የአካባቢ ስሱ"] },
       },
    { class_id: 11, crop: "Enset", crop_am: "እንሰት", name: "Enset Black Leaf Spot", name_am: "የእንሰት ጥቁር ቅጠል ነጠብጣብ", pathogen: "Pseudocercospora spp.", icon: "⚫", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Dark brown to black soot-like spots scattered across the leaf surface; affected leaves dry prematurely.", visual_pattern_am: "በቅጠል ላይ የሚታዩ ጥቁር እንደ ሻጋታ ነጠብጣቦች፤ የተጠቁ ቅጠሎች በእጅጉ ድርቅ።", advice: ["Remove and destroy heavily spotted leaves to reduce spore spread.", "Improve airflow by removing weeds and overcrowded plants.", "Where available, apply a registered fungicide at the first sign of spots."], advice_am: ["በእጅጉ የተጠቁ ቅጠሎችን ይቁረጡና ያስወግዱ።", "የአየር ፍሰትን በማሻሻል ሽፋንን ይቀንሱ።", "ካለ የተፈቀደ ፈንገስ መንከር በመጀመሪያው ምልክት ላይ ይጠቀሙ።"], prevention_tips: ["Space plants widely and keep the field weed-free for good airflow.", "Avoid working among wet leaves, which spreads spores."], prevention_tips_am: ["ተክሎችን በስፋት ያርቁ እና ጥሩ የአየር ፍሰት ይጠብቁ።", "ቅጠሎች ሲኖሩ በርቷቸው ላይ አይሰሩ።"], seasonality: "Humid periods and dense canopy; worst in the main rainy season.", seasonality_am: "በእርጥበት ወቅት እና ጥቅጥቅ ያለ ሽፋን፤ በዋናው የዝናብ ወቅት ከባድ።", home_remedies: ["Traditional: sprinkle wood ash around the base of plants to suppress spore spread.", "Easy: pick off and burn the most-spotted leaves by hand — free and immediate."], home_remedies_am: ["ባህላዊ፡ የእንጨት አመድ ዙሪያቸውን ይረጩ።", "ቀላል፡ በእጅ የተጠቁ ቅጠሎችን ቆርጠው ያቃጥሉ።"], landraces: { resistant: ["Lemat", "Beshute"], susceptible: ["Ageremremat"] }, landraces_am: { resistant: ["ለማት", "በሹቴ"], susceptible: ["አገረምረማት"] },
       },
    { class_id: 12, crop: "Enset", crop_am: "እንሰት", name: "Enset Leaf Tip Dieback", name_am: "የእንሰት ቅጠል ጫፍ መድረቅ", pathogen: "Nutritional / water stress (abiotic)", icon: "🟤", risk: { level: "low", label: "🟢 Low Risk", label_am: "🟢 ዝቅተኛ አደጋ", color: "#16a34a" },
      visual_pattern: "Yellowing, browning, and drying that starts at leaf tips and margins and moves inward; no bacterial ooze.", visual_pattern_am: "ከቅጠል ጫፍ እና ጠርዝ ጀምሮ የሚታይ ቢጫነት፣ ቡናማነት እና ድርቅ፤ የባክቴሪያ ፈሳሽ የለም።", advice: ["Water deeply during dry spells and mulch around the base to hold moisture.", "Apply well-rotted manure or compost to correct nutrient shortfall.", "Rule out bacterial wilt: check a cut pseudostem for yellow ooze before treating as stress."], advice_am: ["በድርቅ ወቅት ውሃ ይጠጡ እና ዙሪያቸውን በማቆያ እርጥበት ይጠብቁ።", "የተበሰበሰ ሌምም ወይም ኮማፖስት በመጨመር የአፈር ንጥረ ነገር ያስተካክሉ።", "የባክቴሪያ ዊልትን ያስወግዱ፡ ግንዱን ቆርጠው ቢጫ ፈሳሽ መኖሩን ያረጋግጡ።"], prevention_tips: ["Mulch and irrigate before the dry season begins.", "Keep enset plots fertile — stressed plants show worse dieback."], prevention_tips_am: ["ከደረቅ ወቅት መጀመሪያ በፊት ማቆያ ይጣሉ እና ያጠጡ።", "የእንሰት እርሻዎችን ለም ይጠብቁ።"], seasonality: "Most visible in the dry season (December–February) and on poor soils.", seasonality_am: "በደረቅ ወቅት (ታኅሣሥ–የካቲት) እና በድህን አፈር ላይ ከባድ።", home_remedies: ["Traditional: work burnt ash and manure into the soil around affected mats.", "Easy: grass mulch around the base keeps roots cool and moist — costs nothing."], home_remedies_am: ["ባህላዊ፡ አመድ እና ሌምም በአፈሩ ውስጥ ይቀላቅሉ።", "ቀላል፡ ሣር በስሩ ማሰራጨት ሙቀትንና እርጥበትን ይጠብቃል።"], landraces: { resistant: [], susceptible: [] }, landraces_am: { resistant: [], susceptible: [] },
       },
    { class_id: 13, crop: "Coffee", crop_am: "ቡና", name: "Coffee Leaf Miner", name_am: "የቡና ቅጠል ቆፋሪ", pathogen: "Leucoptera coffeella (moth larva, pest)", icon: "🐛", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Dry, whitish winding tunnels and papery window-like patches inside leaves; early leaf drop in heavy attacks.", visual_pattern_am: "በቅጠል ውስጥ ነጭ ጠማማ መቆፈሪያ መስመሮች እና እንደ መስኮት ነጠብጣቦች፤ በእጅጉ ሲያዝ ቅጠል መንቀል።", advice: ["Collect and destroy severely mined leaves along with the larvae inside.", "Encourage natural enemies by avoiding broad-spectrum insecticide sprays.", "In heavy outbreaks, apply a registered, targeted insecticide following the label."], advice_am: ["በእጅጉ የተጠቁ ቅጠሎችን ከእንሰሳው ጋር ይሰብስቡና ያጠፉ።", "የተፈጥሯዊ ጠላቶቻቸውን ለመጠበቅ ሰፊ የመንከር መርዝ አይጠቀሙ።", "በእጅጉ ሲያዝ የተፈቀደ ንክክል መንከር በመመሪያው መሠረት ይጠቀሙ።"], prevention_tips: ["Maintain balanced shade to discourage moth population booms.", "Remove fallen mined leaves from around the trees."], prevention_tips_am: ["መጠነኛ ጥላ ይጠብቁ።", "የወደቁ የተጠቁ ቅጠሎችን ከዛፎቹ ዙሪያ ያስወግዱ።"], seasonality: "Warm, dry periods between rainy seasons.", seasonality_am: "በዝናብ ዘመናት መካከል ባሉ ሞቅ ያሉ ደረቅ ጊዜያት።", home_remedies: ["Traditional: hand-pick mined leaves early in the morning when larvae sit near the surface.", "Easy: a neem leaf or seed soak sprayed weekly deters egg-laying moths."], home_remedies_am: ["ባህላዊ፡ ጠዋት ጠዋት የተቆፈሩ ቅጠሎችን በእጅ ይንቀሉ።", "ቀላል፡ የኦዲንጂ ቅጠል መርጫ በሳምንት አንድ ጊዜ ይርጩ።"], landraces: { resistant: ["No known resistant varieties"], susceptible: ["Geisha", "Caturra"] }, landraces_am: { resistant: ["የታወቀ ተከላካይ ዝርያ የለም"], susceptible: ["ጌሻ", "ካቱራ"] },
       },
    { class_id: 14, crop: "Coffee", crop_am: "ቡና", name: "Coffee Sooty Mold", name_am: "የቡና ጥቁር ሻጋታ", pathogen: "Capnodium spp.", icon: "🖤", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Black, charcoal-like powdery coating smothering the leaf surface; rubs off easily and often follows sap-sucking insects.", visual_pattern_am: "የቅጠል ላይን የሚሸፍን ጥቁር እንደ ከሰል ዱቄት፤ በቀላሉ ይሰርዛል።", advice: ["Control the sap-sucking insects (scale, mealybug, aphids) whose honeydew feeds the mold.", "Wash or brush off light mold coatings; leaves recover once the source is gone.", "Prune dense branches to improve light and airflow."], advice_am: ["ሻጋታውን የሚመግቡ እንስሳትን ይቆጣጠሩ።", "ቀላል ሻጋታ በማጠብ ወይም በመሸራበት ያስወግዱ።", "ቅርንጫፎችን በመከርከም ብርሃንና የአየር ፍሰት ያሳድጉ።"], prevention_tips: ["Monitor for scale insects and mealybugs — controlling them prevents the mold.", "Avoid excessive shade; keep trees vigorous."], prevention_tips_am: ["የመጠቀሻ እንስሳትን ይቆጣጠሩ።", "ብዙ ጥላ ያስወግዱ፤ ዛፎችን ጤናማ ያድርጉ።"], seasonality: "Follows honeydew-producing insects; peaks in warm, still weather.", seasonality_am: "የንብ ማምረቻ እንስሳትን ተከትሎ፤ በሙቅ አየር ከፍ ይላል።", home_remedies: ["Traditional: soapy water from local bar soap sprayed on coated leaves loosens the mold.", "Easy: wipe coated leaves with a damp cloth on young trees — immediate relief."], home_remedies_am: ["ባህላዊ፡ የሳሙና ውሃ በመርጨት ሻጋታውን ያላቅቁ።", "ቀላል፡ በእርጥብ ጨርቅ ይሸርብሸሩ።"], landraces: { resistant: ["No known resistant varieties"], susceptible: ["Geisha"] }, landraces_am: { resistant: ["የታወቀ ተከላካይ ዝርያ የለም"], susceptible: ["ጌሻ"] },
       },
    { class_id: 15, crop: "Maize", crop_am: "በቆሎ", name: "Common Corn Rust", name_am: "የበቆሎ ጉፍፊ አበሳ / ዝገት", pathogen: "Puccinia sorghi", icon: "🟠", risk: { level: "moderate", label: "🟡 Moderate Risk", label_am: "🟡 መካከለኛ አደጋ", color: "#f9a825" },
      visual_pattern: "Small, circular orange to cinnamon-brown powdery pustules scattered on both leaf surfaces; leaves a powdery residue on fingers.", visual_pattern_am: "ትንንሽ ክብ ብርቱካንማ-ቡናማ ዱቄታማ እብጥቦች በሁለቱም የቅጠል ገጽ።", advice: ["Scout weekly; if pustules cover the upper leaves before grain fill, apply a registered fungicide.", "Bury or compost maize residue after harvest to break the disease cycle.", "Plant tolerant hybrids next season in affected fields."], advice_am: ["በሳምንት አንድ ጊዜ ይመርምሩ፤ ስርጭት ሲሰፋ ፈንገስ መንከር ይጠቀሙ።", "ከመከር በኋላ የበቆሎ ቅሪትን ቆፍረው ይቅበሩ።", "በሚቀጥለው ወቅት የመከላከያ ዝርያዎችን ይዝሩ።"], prevention_tips: ["Plant early so grain fill finishes before cool, humid rust weather.", "Avoid very dense sowing, which keeps leaves wet."], prevention_tips_am: ["ቀድመው ይዝሩ።", "በጣም ጥቅጥቅ ያለ መዝራት ያስወግዱ።"], seasonality: "Cool nights with dew and moderate humidity — typically mid to late season.", seasonality_am: "ቀዝቃዛ ሌሊቶች እና ጠዋት እርጥበት — በወቅቱ መሃል እና መጨረሻ።", home_remedies: ["Traditional: rotate maize plots away from last season's field — rust spores die without a host.", "Easy: remove and burn the first few rusted plants you find to slow the spread."], home_remedies_am: ["ባህላዊ፡ የበቆሎ እርሻን ከወቅቱ መስክ ያርቁ።", "ቀላል፡ የመጀመሪያዎቹን የተጠቁ ተክሎች ቆርጠው ያቃጥሉ።"], landraces: { resistant: ["Resistant hybrids (check local extension)"], susceptible: ["Local open-pollinated"] }, landraces_am: { resistant: ["ተከላካይ ድብልዎች (የአካባቢ ባለሙያን ይጠይቁ)"], susceptible: ["የአካባቢ ክፍት-የአበባ ዝርያ"] },
       },
    { class_id: 16, crop: "Maize", crop_am: "በቆሎ", name: "Maize Streak Virus (MSV)", name_am: "የበቆሎ ስትሪክ ቫይረስ", pathogen: "Maize streak virus (leafhopper-transmitted)", icon: "🟡", risk: { level: "high", label: "🟠 High Risk", label_am: "🟠 ከፍተኛ አደጋ", color: "#ea580c" },
      visual_pattern: "Long, narrow white to yellow streaks running parallel along the veins; severe stunting and small cobs in young plants.", visual_pattern_am: "ከቅጠል ዙሪያ ጋር በተመሳሳይ አቅጣጫ የሚሮጡ ረጅም ነጭ-ቢጫ መስመሮች፤ ከባድ እድገት ማቆም።", advice: ["Uproot and destroy severely streaked young plants — they never recover.", "Control leafhopper vectors: clear grass weeds in and around the field.", "Plant resistant/tolerant varieties in areas with known MSV history."], advice_am: ["በእጅጉ የተጠቁ ወጣት ተክሎችን ቆፍረው ያስወግዱ — እነሱ አይታከሙም።", "የተቀናጁ ተባዮችን ይቆጣጠሩ፡ በእርሻው ውስጥና ዙሪያ ሣር ያስወግዱ።", "በታሪክ በሚታወቁ አካባቢዎች የመከላከያ ዝርያዎችን ይዝሩ።"], prevention_tips: ["Plant at the start of the rains so seedlings grow quickly past the vulnerable stage.", "Remove volunteer maize and grass weeds that harbour leafhoppers.", "Use certified seed of tolerant varieties."], prevention_tips_am: ["በዝናብ መጀመሪያ ላይ ይዝሩ።", "የተረፉ ተክሎችንና ሣር ያስወግዱ።", "የመከላከያ ዝርያዎችን ይጠቀሙ።"], seasonality: "Early season, when leafhoppers move from grasses to young maize.", seasonality_am: "የመጀመሪያው ወቅት፣ ተቀናጆች ከሣር ወደ ወጣት በቆሎ ሲቀየሩ።", home_remedies: ["Traditional: early removal of streaked plants protects the rest of the stand.", "Easy: keep field edges clear of tall grass where leafhoppers breed."], home_remedies_am: ["ባህላዊ፡ የተጠቁ ተክሎችን በጊዜው ማስወገድ ቀሪውን ይጠብቃል።", "ቀላል፡ የመስክ ጠርዞችን ከሣር ንጹህ ያድርጉ።"], landraces: { resistant: ["Resistant hybrids (check local extension)"], susceptible: ["Local susceptible"] }, landraces_am: { resistant: ["ተከላካይ ድብልዎች (የአካባቢ ባለሙያን ይጠይቁ)"], susceptible: ["የአካባቢ ስሱ"] },
       }
  ]
};

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
// FIX (unified dashboard): dashTab removed — the dashboard is a single disease-list view.
let dashCropFilter = 'All';         // dashboard crop filter: 'All' | 'Enset' | 'Coffee' | 'Maize' (new)
let dashDataLoading = false;        // dedupe concurrent loadDashData() calls (FIX)
let dashDataWarned = false;         // guard a single offline toast, not duplicates (FIX)
// FIX (disease detail): view-stack + detail-view state (kept so back returns correctly)
let viewStack = [];                 // array of 'home' | 'dashboard' | 'history' you can return to
let currentDetail = null;           // { classId, fromHistory, historyRecord } for re-render on lang switch
let currentDetailScroll = 0;        // remember scroll when leaving dashboard/history
let cameraStream = null;            // active getUserMedia stream
let cameraActive = false;           // whether the viewfinder is live
let cameraStable = false;           // stability detector state
let cameraStableSince = 0;          // timestamp when stability was achieved
let cameraLight = 'checking';       // 'good' | 'dark' | 'bright' | 'checking'
let cameraRaf = null;               // requestAnimationFrame handle
let cameraLoopCanvas = null;        // reused downsampling canvas (bug #8)
let cameraPrevFrame = null;         // previous downscaled frame for motion detection
let cameraMotion = 0;               // current motion score 0..1
let cameraSupported = false;        // whether getUserMedia is available
let mlWorker = null;                // dedicated ML worker (heuristic offload)
let mlWorkerReady = false;          // whether the worker responded successfully
let runInferenceSeq = 0;            // monotonic id to discard stale worker results (audit fix)
let diagnosisRunning = false;       // re-entrancy guard for the Analyze button (audit fix)
let historyEpoch = 0;               // bumped on history clear; invalidates in-flight logging (audit fix)
let inferenceCanvas = null;         // reused canvas for getImageData() (audit fix)

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
const viewDetail = el('view-detail');
const viewHistory = el('view-history');
const detailBackBtn = el('detailBackBtn');
const detailBackText = el('detailBackText');
const detailPhoto = el('detailPhoto');
const detailName = el('detailName');
const detailCrop = el('detailCrop');
const detailRisk = el('detailRisk');
const detailSections = el('detailSections');
const detailScanMeta = el('detailScanMeta');
const detailScanDate = el('detailScanDate');
const detailScanConfidence = el('detailScanConfidence');
const detailScanResult = el('detailScanResult');
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
    // Dashboard crop filter + accordion sub-sections (bilingual, new)
    filter_all: 'All',
    crop_enset: 'Enset',
    crop_coffee: 'Coffee',
    crop_maize: 'Maize',
    sub_symptoms: 'Symptoms',
    sub_prevention: 'Prevention',
    sub_treatment: 'Treatment',
    sub_remedies: 'Traditional Remedies',
    sub_season: 'Seasonality',
    sub_varieties: 'Resistant Varieties',
    no_varieties: 'None listed',
    none: 'None',
    no_known_varieties: 'No known resistant varieties',
    diseases_title: 'Disease Dashboard',
    detail_back: 'Back',
    detail_scanned_on: 'Scanned on',
    detail_confidence: 'Confidence',
    detail_result: 'Result',
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
    // የዳሽቦርድ የሰብል ማጣሪያ እና አኞርዲዮን ክፍሎች (አዲስ)
    filter_all: 'ሁሉም',
    crop_enset: 'እንሰት',
    crop_coffee: 'ቡና',
    crop_maize: 'በቆሎ',
    sub_symptoms: 'ምልክቶች',
    sub_prevention: 'መከላከያ',
    sub_treatment: 'ህክምና',
    sub_remedies: 'ባህላዊ መፍትሄዎች',
    sub_season: 'ወቅት',
    sub_varieties: 'ተከላካይ ዝርያዎች',
    no_varieties: 'የለም',
    none: 'የለም',
    no_known_varieties: 'የታወቀ ተከላካይ ዝርያ የለም',
    diseases_title: 'የበሽታ ዳሽቦርድ',
    detail_back: 'ተመለስ',
    detail_scanned_on: 'የተመረመረበት',
    detail_confidence: 'እርግጠኝነት',
    detail_result: 'ውጤት',
    start_scan_title: 'ለመመርመር ዝግጁ ነዎት?',
    start_scan_body: 'ፎቶ ያንሱ ወይም ከጋለሪዎ ይምረጡ ለመጀመር።',
    start_scan_btn: 'ምርመራ ይጀምሩ'
  }
};
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

  // FIX (unified dashboard): Dashboard now has a single disease-list view — no
  // `.dash-tab` buttons remain, so the tab-switching listener is removed. The
  // `dashBody` delegated click handler (added in bindEvents) handles the
  // accordion and sub-section toggles via data-action/data-key.

  // Crop filter chips — clicking re-renders the single disease-list dashboard.
  const cropFilterBar = el('cropFilter');
  if (cropFilterBar) cropFilterBar.addEventListener('click', (e) => {
    const chip = e.target.closest('.crop-chip');
    if (!chip) return;
    dashCropFilter = chip.dataset.crop || 'All';
    buildCropFilter();
    renderDashboard();
  });

  // Accordion toggles (new) — delegated via data-action/data-key on dashBody so
  // they survive innerHTML re-renders (per spec): card header toggles the whole
  // card; sub-section toggles expand/collapse each section independently.
  if (dashBody) dashBody.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.getAttribute('data-action');
    if (action === 'open-detail') {
      const classId = parseInt(btn.getAttribute('data-key'), 10);
      if (!isNaN(classId)) openDiseaseDetail(classId, false, null);
    }
  });

  // Start Scan button → opens the photo source chooser
  if (startScanBtn) startScanBtn.addEventListener('click', () => {
    hide(startScan);
    show(cameraCard);
    openChooser();
  });

  // Camera viewfinder controls: the photo source chooser opens ONLY via the
  // Capture / Gallery buttons (bug #7 — no click-through on the camera card).
  if (chooser) chooser.addEventListener('click', closeChooserOnBackdrop);

  // Keyboard accessibility: Escape closes the chooser sheet (audit fix).
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && chooser && !chooser.hidden) hide(chooser);
  });

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
  // Keyboard accessibility: focus the first option when the sheet opens.
  if (chooserCameraBtn) chooserCameraBtn.focus();
}

function closeChooserOnBackdrop(e) {
  if (e.target === chooser) hide(chooser);
}

/** Load the disease catalog with a 10-second timeout (bugs #5, #17).
 *  Falls back to the built-in minimal catalog; if even that fails,
 *  warns the user via toast. */
async function loadCatalog() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CATALOG_TIMEOUT_MS);
  try {
    const res = await fetch(CATALOG_URL, { signal: controller.signal });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (!data || !Array.isArray(data.classes) || data.classes.length === 0) {
      throw new Error('Invalid catalog structure');
    }
    return data;
  } catch (err) {
    console.warn('Catalog fetch failed/timeout — using built-in minimal catalog:', err.message);
    if (MINIMAL_CATALOG && Array.isArray(MINIMAL_CATALOG.classes)) {
      // Never degrade silently — tell the user (audit fix).
      showToast(lang === 'am'
        ? 'የተገደበ ከመስመር ውጭ ውሂብ በአገልግሎት ላይ — ሙሉ የበሽታ ዝርዝሮችን ለማግኘት እባክዎ እንደገና ይገናኙ።'
        : 'Limited offline data in use — reconnect to load full disease details.');
      return MINIMAL_CATALOG;
    }
    // Should never happen (built-in catalog is static), but guard anyway.
    showToast(lang === 'am'
      ? 'እባክዎ የበይነመረብ ግንኙነትዎን ያስተካክሉ የቅርብ የበሽታ መረጃ ለማግኘት።'
      : 'Please fix your internet connection to get the latest disease data.');
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/* ============================================================
   INITIALIZATION
   ============================================================ */
document.addEventListener('DOMContentLoaded', async () => {
  // FIX (offline): register the service worker FIRST, before any awaits -
  // the catalog fetch has a 10 s timeout and would otherwise delay or
  // entirely skip SW registration on slow/offline first loads, which is
  // the main cause of "app doesn't work offline".
  registerServiceWorker();

  // FIX (install button): when opened already-installed (standalone), never
  // show the install UI even if a stale beforeinstallprompt fires later.
  if (isStandalone()) hideInstallUI();

  catalog = await loadCatalog();

  // Load saved language preference
  lang = (typeof localStorage !== 'undefined' && localStorage.getItem('ensetscan-lang')) || 'en';
  applyLang();

  // Load national dashboard data (non-blocking)
  // Non-blocking: the dashboard view lazy-loads and renders when opened.
  // An await here would stall DB init, event binding and SW registration
  // on a slow/stalled connection (audit fix).
  loadDashData().catch(() => {});

  await initDatabase();
  bindEvents();
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

  // Language sync (FIX): when switching language, instantly re-render whatever
  // view is visible — dashboard, history, or the disease detail view.
  if (viewDashboard && !viewDashboard.hidden) {
    buildCropFilter();
    renderDashboard();
  } else if (viewHistory && !viewHistory.hidden) {
    openHistory();
  } else if (viewDetail && !viewDetail.hidden && currentDetail) {
    detailBackText.textContent = t.detail_back;
    renderDiseaseDetail(currentDetail.classId, currentDetail.fromHistory, currentDetail.historyRecord);
  }
}

/** Localized helper: pick the Amharic or English value from a class field.
 *  Falls back to the English value (with a console warning) when the
 *  Amharic translation is missing (bugs #2, #13). */
function loc(cls, enKey, amKey) {
  if (lang === 'am') {
    if (cls[amKey] !== undefined && cls[amKey] !== null) return cls[amKey];
    console.warn('Missing Amharic translation for "' + amKey + '" — falling back to English.');
  }
  return cls[enKey];
}

/** HTML-escape a data string before it is interpolated into innerHTML
 *  (audit fix: XSS hardening for catalog/dashboard-derived content). */
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
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

/** True when the app is running installed/standalone (home-screen or browser). */
function isStandalone() {
  if (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) return true;
  if (navigator && navigator.standalone === true) return true; // iOS Safari
  return false;
}

/** Hide the install header button and banner (used after install / dismissal). */
function hideInstallUI() {
  installBtn.hidden = true;
  if (installBanner) installBanner.hidden = true;
}

function installApp() {
  if (!deferredInstallPrompt) {
    // No install prompt available (e.g. already installed or unsupported).
    // Keep the banner visible but do nothing harmful.
    console.info('Install prompt not available.');
    return;
  }
  deferredInstallPrompt.prompt();
  // FIX (install button): hide the UI immediately on click — before the
  // prompt resolves — so the button never lingers after the user installs.
  hideInstallUI();
  deferredInstallPrompt.userChoice.then(() => {
    // Hide regardless of outcome (accepted or dismissed), per spec.
    deferredInstallPrompt = null;
    hideInstallUI();
  });
}

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredInstallPrompt = e;
  // FIX (install button): only show when NOT already installed/standalone.
  if (isStandalone()) {
    hideInstallUI();
  } else {
    installBtn.hidden = false;
    if (installBanner) installBanner.hidden = false;
  }
});

window.addEventListener('appinstalled', () => {
  deferredInstallPrompt = null;
  hideInstallUI();
});

/** Service worker registration — called as early as possible (the very first
 *  line of DOMContentLoaded) so the cache install runs immediately, making the
 *  app usable offline on the very next visit. (FIX: offline mode) */
function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch((err) => {
      console.warn('Service worker registration failed:', err);
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
  // Re-entrancy guard: `cameraActive` is only set AFTER the async getUserMedia
  // resolves, so it alone cannot prevent concurrent starts (audit fix).
  if (cameraActive || cameraStarting) return;
  cameraStarting = true;

  cameraSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  if (!cameraSupported) {
    cameraStarting = false;
    showCameraError('unsupported');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    // Release any stream that slipped through before assigning (audit fix:
    // overwriting cameraStream without stopping tracks leaks the hardware lock).
    if (cameraStream) cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = stream;
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
    console.warn('Camera unavailable:', err && err.name, err && err.message);
    showCameraError(err && err.name);
  } finally {
    cameraStarting = false;
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

/** Show the camera-unavailable fallback UI with a clear, non-technical
 *  message matched to the failure cause (audit fix). */
function showCameraError(kind) {
  if (!cameraError) return;
  cameraError.hidden = false;
  if (cameraVideo) cameraVideo.hidden = true;
  if (reticle) reticle.hidden = true;
  if (cameraStatus) cameraStatus.hidden = true;
  if (captureBtn) captureBtn.disabled = true;
  if (cameraErrorText) {
    const am = lang === 'am';
    let msg;
    switch (kind) {
      case 'NotAllowedError':
        msg = am
          ? 'የካሜራ ፈቃድ ታግዷል። እባክዎ ከአድራሻ አሞሌው ላይ 🔒 ምልክቱን ተጠቅመው ካሜራውን ይፍቀዱ እና እንደገና ይሞክሩ።'
          : 'Camera access was blocked. Tap the 🔒 icon in your browser\u2019s address bar, allow camera, and try again.';
        break;
      case 'NotFoundError':
      case 'OverconstrainedError':
        msg = am
          ? 'ካሜራ አልተገኘም — እባክዎ ከፎቶ ባህሪው ይምረጡ።'
          : 'No camera found — please use \u201cChoose from Gallery\u201d instead.';
        break;
      case 'NotReadableError':
        msg = am
          ? 'ካሜራው በሌላ መተግበሪያ ተይዟል — ያንን መተግበሪያ ዘግተው እንደገና ይሞክሩ።'
          : 'The camera is in use by another app — close it and try again.';
        break;
      case 'unsupported':
        msg = am
          ? 'ይህ አሳሽ የቀጥታ ካሜራ አይደግፍም — እባክዎ ከፎቶ ባህሪው ይምረጡ።'
          : 'This browser doesn\u2019t support live camera — please use \u201cChoose from Gallery\u201d.';
        break;
      default:
        msg = am
          ? 'ካሜራውን መጀመር አልተቻለም — እባክዎ እንደገና ይሞክሩ ወይም ከፎቶ ባህሪው ይምረጡ።'
          : 'Couldn\u2019t start the camera — please try again or use \u201cChoose from Gallery\u201d.';
    }
    cameraErrorText.textContent = msg;
  }
}

/** Main rAF loop: sample light + motion, update chips and reticle. */
function cameraLoop() {
  // Hard exit when the camera is stopped — a stopped loop must never
  // reschedule itself (audit fix: zombie rAF loop defense).
  if (!cameraActive) return;
  if (!cameraVideo || cameraVideo.readyState < 2) {
    cameraRaf = requestAnimationFrame(cameraLoop);
    return;
  }

  // Downscale to a tiny working canvas for cheap pixel analysis.
  // The canvas is created once and reused across frames (bug #8).
  const w = 64, h = 48;
  if (!cameraLoopCanvas) {
    cameraLoopCanvas = document.createElement('canvas');
    cameraLoopCanvas.width = w;
    cameraLoopCanvas.height = h;
  }
  const canvas = cameraLoopCanvas;
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
  if (typeof Worker === 'undefined') return null;
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
    return mlWorker;
  } catch (err) {
    mlWorker = null;
    mlWorkerReady = false;
    return null;
  }
}

/**
 * Run inference, preferring the dedicated worker when available.
 * Falls back to main-thread heuristic if the worker is unavailable
 * or fails to respond.
 */
function runInference(img) {
  return new Promise((resolve) => {
    const reqId = ++runInferenceSeq; // identifies THIS scan (audit fix: stale-result race)
    const fallback = () => {
      const r = analyzeHeuristic(img);
      resolve({ probs: r.probs, engine: 'heuristic', elapsedMs: r.elapsedMs, unsupported: false });
    };

    if (!mlWorker || !mlWorkerReady) {
      fallback();
      return;
    }

    // Extract raw pixels at a reasonable working resolution to send to the worker.
    let { data, width, height } = getImageData(img, 256);

    const timeout = setTimeout(() => {
      mlWorkerReady = false;
      data = null; // release the large buffer (bug #9)
      // Terminate and rebuild the stalled worker so its late result can never
      // be attributed to a later scan (audit fix: stale-message race).
      try { mlWorker.terminate(); } catch (e) { /* ignore */ }
      mlWorker = initMLWorker() || null;
      fallback();
    }, 3000);

    mlWorker.onmessage = (e) => {
      if (e.data && e.data.type === 'result') {
        if (e.data.reqId !== reqId) return; // stale result from a previous scan — discard
        clearTimeout(timeout);
        mlWorkerReady = true;
        data = null; // release the large buffer (bug #9)
        resolve({
          probs: e.data.probs,
          engine: e.data.engine || 'heuristic',
          elapsedMs: e.data.elapsedMs,
          unsupported: !!e.data.unsupported // Safari / no OffscreenCanvas (bug #1)
        });
      }
    };

    mlWorker.postMessage({ type: 'infer', reqId, imageData: data, width, height });
    data = null; // buffer copied by structured clone — free it (bug #9)
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
 * Pixel extraction happens here (main thread); the classification rules
 * live in the shared heuristic.js (bug #4/#15) via analyzeHeuristicFromPixels().
 */
function analyzeHeuristic(img) {
  const t0 = performance.now();
  const { data } = getImageData(img, HEURISTIC_SIZE);
  const probs = analyzeHeuristicFromPixels(data);
  return { probs, elapsedMs: performance.now() - t0 };
}

/* ---------------- Image Utilities ---------------- */

/** Center-square-crop an image and return its pixel data at `size`. */
function getImageData(img, size) {
  // Reused across scans — avoids per-inference canvas allocation (audit fix).
  if (!inferenceCanvas) inferenceCanvas = document.createElement('canvas');
  const canvas = inferenceCanvas;
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
  // Drop the <img> reference BEFORE revoking so no broken-image flash (audit fix).
  currentImage = null;
  previewImg.removeAttribute('src');
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
}

/* ============================================================
   DIAGNOSIS PIPELINE
   ============================================================ */
async function runDiagnosis() {
  // Re-entrancy guard: ignore taps while a scan is already running (audit fix:
  // spammed Analyze button launched concurrent pipelines and logged duplicates).
  if (!currentImage || diagnosisRunning) return;
  diagnosisRunning = true;
  const analyzeBtn = el('analyzeBtn');
  if (analyzeBtn) analyzeBtn.disabled = true;
  try {
    await runDiagnosisInner();
  } finally {
    diagnosisRunning = false;
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

async function runDiagnosisInner() {
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
  let unsupportedBrowser = false; // bug #1 — worker could not analyze (no OffscreenCanvas)
  const epochAtScan = historyEpoch; // detect history clears during this scan (audit fix)

  try {
    if (engine === 'neural') {
      probs = predictWithModel(currentImage).probs;
    } else {
      // Prefer the dedicated worker; fall back to main-thread heuristic.
      const r = await runInference(currentImage);
      probs = r.probs;
      engine = r.engine;
      unsupportedBrowser = !!r.unsupported;
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

  // If the user cleared history while this scan was running, respect their
  // intent and skip logging (audit fix: clear-vs-scan race).
  if (epochAtScan === historyEpoch) {
    logScan(record);
    updateAppStats(record.timestamp);
  }

  if (confirmed) {
    lastResult = { cls: catalog.classes[topIdx], conf: topConf, engine, inconclusive: false };
    renderResult(catalog.classes[topIdx], topConf, engine);
  } else {
    lastResult = { conf: topConf, engine, inconclusive: true };
    // Bug #1: browser lacks OffscreenCanvas — show the dedicated message.
    const unsupportedMsg = lang === 'am'
      ? 'የድረ-አሳሽዎ ከመስመር ውጭ ትንተናን አይደግፍም — እባክዎ Chrome ይጠቀሙ።'
      : 'Your browser doesn\u2019t support offline analysis \u2013 please use Chrome';
    if (unsupportedBrowser) {
      showToast(unsupportedMsg);
    }
    renderInconclusive(topConf, engine, unsupportedBrowser ? unsupportedMsg : null);
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
    appendResultBlock(resultCard, t.local_names, esc(cls.local_names.join(' · ')));
  }

  // Regional statistics
  if (cls.regional_stats) {
    appendResultBlock(
      resultCard,
      t.regional_title,
      esc(lang === 'am' ? cls.regional_stats_am : cls.regional_stats),
      'regional'
    );
  }

  // Traditional practices
  if (cls.traditional_practices) {
    appendResultBlock(
      resultCard,
      t.traditional_title,
      esc(lang === 'am' ? cls.traditional_practices_am : cls.traditional_practices),
      'traditional'
    );
  }

  // Landrace recommendation banner (enset classes)
  if (cls.landraces && cls.landraces.resistant) {
    const lr = lang === 'am' ? cls.landraces_am : cls.landraces;
    const html =
      '<strong>' + t.landrace_resistant + '</strong>' + esc(lr.resistant.join(', ')) +
      '<br /><strong>' + t.landrace_susceptible + '</strong>' + esc(lr.susceptible.join(', '));
    appendResultBlock(resultCard, t.landrace_title, html, 'landrace');
  }

  // Resistant accessions (CBD)
  if (cls.resistant_accessions && cls.resistant_accessions.length) {
    appendResultBlock(
      resultCard,
      t.resistant_accessions,
      esc(cls.resistant_accessions.join(', ')),
      'accessions'
    );
  }

  // Tool sanitation priority (Enset BW)
  if (cls.sanitation) {
    appendResultBlock(
      resultCard,
      t.sanitation_title,
      esc(lang === 'am' ? cls.sanitation.priority_am : cls.sanitation.priority),
      'sanitation'
    );
  }

  // Canopy advisor (coffee under enset shade)
  if (cls.canopy_advisor) {
    appendResultBlock(
      resultCard,
      t.canopy_title,
      esc(lang === 'am' ? cls.canopy_advisor_am : cls.canopy_advisor),
      'canopy'
    );
  }

  // Seasonality
  if (cls.seasonality) {
    appendResultBlock(
      resultCard,
      t.seasonality_title,
      esc(lang === 'am' ? cls.seasonality_am : cls.seasonality),
      'seasonality'
    );
  }

  // Transmission
  if (cls.transmission) {
    appendResultBlock(
      resultCard,
      t.transmission_title,
      esc(lang === 'am' ? cls.transmission_am : cls.transmission),
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
      '<span class="spread-badge spread-' + esc(cls.spread_rate) + '">' + esc(spreadLabel) + '</span>',
      'spread'
    );
  }

  // Yield impact
  if (cls.yield_impact) {
    appendResultBlock(
      resultCard,
      t.yield_impact_title,
      esc(lang === 'am' ? cls.yield_impact_am : cls.yield_impact),
      'yield'
    );
  }

  // Monitoring frequency
  if (cls.monitoring_frequency) {
    appendResultBlock(
      resultCard,
      t.monitoring_title,
      esc(lang === 'am' ? cls.monitoring_frequency_am : cls.monitoring_frequency),
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
    const html = '<ul class="prevention-list">' + tips.map((tip) => '<li>' + esc(tip) + '</li>').join('') + '</ul>';
    appendResultBlock(resultCard, t.prevention_title, html, 'prevention');
  }

  // Traditional & easy home remedies (prominent)
  if (cls.home_remedies && cls.home_remedies.length) {
    const remedies = lang === 'am' && cls.home_remedies_am ? cls.home_remedies_am : cls.home_remedies;
    const html = '<ul class="remedy-list">' + remedies.map((r) => '<li>' + esc(r) + '</li>').join('') + '</ul>';
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
  label.textContent = t.food_loss_plants + ': ';
  const input = document.createElement('input');
  input.type = 'number';
  input.className = 'foodLossCount';
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

function renderInconclusive(conf, engine, reasonMsg) {
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

  visualMatch.textContent = reasonMsg || (lang === 'am'
    ? 'ምስሉ በጣም የደበዘዘ፣ ብርሀኑ ደካማ ወይም የቅጠሉ ንድፍ አሻሚ ስለሆነ አስተማማኝ ምርመራ ማድረግ አይቻልም።'
    : 'The image is too blurry, poorly lit, or the leaf pattern is ambiguous for a reliable diagnosis.');

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
  try {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).add(record);
    tx.onerror = () => handleStorageError();
  } catch (err) {
    handleStorageError(err);
  }
}

function getAllScans() {
  return new Promise((resolve) => {
    if (!db) return resolve([]);
    try {
      const tx = db.transaction(DB_STORE, 'readonly');
      const req = tx.objectStore(DB_STORE).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => resolve([]);
    } catch (err) {
      handleStorageError(err);
      resolve([]);
    }
  });
}

function clearAllScans() {
  return new Promise((resolve) => {
    if (!db) return resolve();
    try {
      const tx = db.transaction(DB_STORE, 'readwrite');
      tx.objectStore(DB_STORE).clear();
      tx.oncomplete = () => {
        historyEpoch++; // invalidate any in-flight scan logging (audit fix)
        resolve();
      };
      tx.onerror = () => resolve();
    } catch (err) {
      handleStorageError(err);
      resolve();
    }
  });
}

/** Handle IndexedDB failures (e.g. quota exceeded) with a prominent toast
 *  offering to clear the scan history (bugs #3, #12). */
function handleStorageError(err) {
  if (err) console.warn('IndexedDB error:', err);
  showToast(
    lang === 'am'
      ? 'የስካን ታሪክ ማከማቻ ሙሉ ነው — ታሪኩን ያጽዱ።'
      : 'Scan history storage is full — please clear your history.',
    lang === 'am' ? 'ታሪክ አጽዳ' : 'Clear History',
    () => { clearAllScans().then(() => openHistory()); }
  );
}

/* ---------------- Toast notifications ---------------- */

/** Show a toast in #toastContainer with an optional action button. */
function showToast(message, actionLabel, actionFn) {
  const container = el('toastContainer');
  if (!container) {
    console.warn('Toast (no container):', message);
    return;
  }
  const toast = document.createElement('div');
  toast.className = 'toast';

  const msg = document.createElement('span');
  msg.className = 'toast-msg';
  msg.textContent = message;
  toast.appendChild(msg);

  if (actionLabel && typeof actionFn === 'function') {
    const btn = document.createElement('button');
    btn.className = 'toast-btn';
    btn.type = 'button';
    btn.textContent = actionLabel;
    btn.addEventListener('click', () => {
      actionFn();
      toast.remove();
    });
    toast.appendChild(btn);
  }

  const close = document.createElement('button');
  close.className = 'toast-close';
  close.type = 'button';
  close.setAttribute('aria-label', 'Dismiss');
  close.textContent = '✕';
  close.addEventListener('click', () => toast.remove());
  toast.appendChild(close);

  container.appendChild(toast);
  // Auto-dismiss after 8 seconds
  setTimeout(() => toast.remove(), 8000);
}

/** Load the national dashboard data (idempotent, lazy-load friendly).
 *  Guarded by a 10 s AbortController so a stalled connection can never
 *  hang app initialization. Retries once on failure and shows a single,
 *  non-duplicate user-friendly toast (FIX). */
async function loadDashData() {
  if (dashData || dashDataLoading) return;
  dashDataLoading = true;
  const attempts = 2; // one initial attempt + one automatic retry
  try {
    for (let attempt = 0; attempt < attempts; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 10000);
      try {
        const dres = await fetch('national-dashboard.json', { signal: controller.signal });
        if (!dres.ok) throw new Error('HTTP ' + dres.status);
        dashData = await dres.json();
        dashDataWarned = false; // recovered — allow a future toast if it fails again
        break;
      } catch (err) {
        console.warn('Failed to load national dashboard (attempt ' + (attempt + 1) + '):', err);
        if (attempt === attempts - 1 && !dashDataWarned) {
          dashDataWarned = true;
          showToast(lang === 'am'
            ? 'የዳሽቦርድ ውሂብ አይገኝም — እባክዎ የበይነመረብ ግንኙነትዎን ያረጋግጡ እና ያድሱ።'
            : 'Dashboard data unavailable. Please check your internet connection and refresh.');
        }
      } finally {
        clearTimeout(timer);
      }
      if (attempt < attempts - 1) await new Promise((res) => setTimeout(res, 300));
    }
  } finally {
    dashDataLoading = false;
  }
}

/* ---------------- View Navigation ---------------- */
function switchView(view) {
  // FIX (disease detail): detail is a stacked view — keep nav highlighting based
  // on the underlying section (home/dashboard/history), not 'detail'.
  const navKey = view === 'detail' ? (viewStack.length ? viewStack[viewStack.length - 1] : 'home') : view;
  document.querySelectorAll('.nav-item').forEach((item) => {
    item.classList.toggle('active', item.dataset.view === navKey);
  });

  // Update header title
  if (viewTitle && view !== 'detail') {
    const titleKey = 'view_' + view;
    viewTitle.textContent = I18N[lang][titleKey] || 'EnsetScan';
  }

  // Show/hide the main views (detail included)
  viewHome.hidden = view !== 'home';
  viewDashboard.hidden = view !== 'dashboard';
  viewHistory.hidden = view !== 'history';
  if (viewDetail) viewDetail.hidden = view !== 'detail';

  // When returning to Home, reset to the welcome screen if no scan is in progress
  if (view === 'home' && !currentImage && !lastResult) {
    hide(cameraCard);
    hide(previewCard);
    hide(loadingCard);
    hide(resultCard);
    show(startScan);
    stopCamera();
  }

  // Render content on demand; lazy-load dashboard data if not yet available.
  if (view === 'dashboard') {
    buildCropFilter();
    if (dashData) {
      renderDashboard();
    } else {
      loadDashData().then(() => renderDashboard());
    }
  } else if (view === 'history') {
    openHistory();
  } else if (view === 'detail' && currentDetail) {
    // Re-render detail content (e.g. after a language switch) without clearing state
    renderDiseaseDetail(currentDetail.classId, currentDetail.fromHistory, currentDetail.historyRecord);
  }

  // FIX (disease detail): scroll behavior — scroll to top on open, restore on back
  if (view === 'home' || view === 'dashboard' || view === 'history') {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } else if (view === 'detail') {
    window.scrollTo({ top: 0 });
  }
}

/* ---------------- Disease Detail View ---------------- */

/** Open the detail view for a disease, tracking the previous view for back. */
function openDiseaseDetail(classId, fromHistory, historyRecord) {
  if (!catalog || !catalog.classes[classId]) return;

  // Remember which view we came from
  currentDetail = { classId, fromHistory: !!fromHistory, historyRecord: historyRecord || null };
  viewStack.push(fromHistory ? 'history' : 'dashboard');

  // Switch to the detail view (renders content via switchView's detail branch)
  switchView('detail');
}

/** Render (or re-render, e.g. on language switch) the detail view content. */
function renderDiseaseDetail(classId, fromHistory, historyRecord) {
  if (!catalog || !catalog.classes[classId] || !viewDetail) return;
  const cls = catalog.classes[classId];
  const t = I18N[lang];
  const risk = cls.risk || {};

  // Name / crop / risk badge
  const name = (lang === 'am' && cls.name_am) ? cls.name_am : cls.name;
  const crop = (lang === 'am' && cls.crop_am) ? cls.crop_am : cls.crop;
  if (detailName) detailName.textContent = name;
  if (detailCrop) detailCrop.textContent = crop + ' · ' + (cls.pathogen || '');

  // Risk badge (colored)
  if (detailRisk) {
    const lvl = risk.level || 'normal';
    detailRisk.className = 'detail-risk risk-' + lvl;
    detailRisk.textContent = (lang === 'am' ? risk.label_am : risk.label) || '';
  }

  // Photo / fallback
  const img = cls.image || 'default-disease.svg';
  if (detailPhoto) {
    detailPhoto.src = img;
    detailPhoto.alt = name;
  }

  // Scan metadata (only from history)
  if (detailScanMeta) {
    if (historyRecord) {
      detailScanMeta.hidden = false;
      if (detailScanDate) detailScanDate.textContent = '📅 ' + (t.detail_scanned_on || 'Scanned on') + ': ' + new Date(historyRecord.timestamp).toLocaleString();
      if (detailScanConfidence) detailScanConfidence.textContent = '📊 ' + (t.detail_confidence || 'Confidence') + ': ' + historyRecord.confidencePct.toFixed(1) + '%';
      if (detailScanResult) detailScanResult.textContent = '🏷️ ' + (t.detail_result || 'Result') + ': ' + name;
    } else {
      detailScanMeta.hidden = true;
    }
  }

  // 6 collapsible sections (expanded by default)
  const sections = [
    { key: 'symptoms', icon: '🔍', label: t.sub_symptoms, items: cls.visual_pattern ? [lang === 'am' ? cls.visual_pattern_am : cls.visual_pattern] : [] },
    { key: 'prevention', icon: '🛡️', label: t.sub_prevention, items: (lang === 'am' ? cls.prevention_tips_am : cls.prevention_tips) || [] },
    { key: 'treatment', icon: '💊', label: t.sub_treatment, items: (lang === 'am' ? cls.advice_am : cls.advice) || [] },
    { key: 'remedies', icon: '🏠', label: t.sub_remedies, items: (lang === 'am' ? cls.home_remedies_am : cls.home_remedies) || [] },
    { key: 'season', icon: '📅', label: t.sub_season, items: (lang === 'am' ? cls.seasonality_am : cls.seasonality) ? [lang === 'am' ? cls.seasonality_am : cls.seasonality] : [] },
    { key: 'varieties', icon: '🌿', label: t.sub_varieties, items: (() => {
        const lr = (lang === 'am' ? cls.landraces_am : cls.landraces) || null;
        return lr && lr.resistant && lr.resistant.length ? lr.resistant : [];
      })()
    }
  ];
  if (detailSections) {
    detailSections.innerHTML = sections.map((s, i) => {
      const list = s.items.length ? s.items.map((x) => '<li>' + esc(x) + '</li>').join('') : '<li>' + esc(t.none) + '</li>';
      return '<div class="detail-section open">' +
        '<button type="button" class="detail-sec-head" data-action="toggle-detail-sec" aria-expanded="true">' +
        '<span>' + s.icon + ' ' + esc(s.label) + '</span><span class="sec-arrow" aria-hidden="true">▼</span></button>' +
        '<div class="detail-sec-body"><ul>' + list + '</ul></div></div>';
    }).join('');
  }
}

/** Back button handler — returns to the previous view (dashboard or history). */
function detailBack() {
  if (!viewStack.length) {
    switchView('dashboard');
    return;
  }
  const prev = viewStack.pop();
  currentDetail = null;
  switchView(prev);
}

// FIX (disease detail): delegated click handler for detail section toggles + back button
if (viewDetail) {
  viewDetail.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (btn) {
      if (btn.getAttribute('data-action') === 'toggle-detail-sec') {
        const sec = btn.closest('.detail-section');
        const open = sec.classList.toggle('open');
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        const arrow = btn.querySelector('.sec-arrow');
        if (arrow) arrow.textContent = open ? '▼' : '▶';
      }
      return;
    }
    if (e.target.closest('#detailBackBtn') || (detailBackBtn && detailBackBtn.contains(e.target))) {
      detailBack();
    }
  });
}
function openDashboard() {
  switchView('dashboard');
}

// FIX (unified dashboard): switchDashTab removed — the dashboard is now a single
// disease-list view driven by renderDashboard(), and the crop filter replaces
// tab switching.

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

/** Map catalog class_id → national-dashboard disease id (for joins). */
const DASH_ID_BY_CLASS = {
  0: 'enset-bacterial-wilt', 8: 'coffee-wilt', 11: 'enset-black-leaf-spot',
  12: 'enset-leaf-tip-dieback', 13: 'coffee-leaf-miner', 14: 'coffee-sooty-mold',
  15: 'corn-rust', 16: 'maize-streak-virus'
};

/** Build the crop filter chip bar (bilingual). Applies to ALL dashboard tabs. */
function buildCropFilter() {
  const bar = el('cropFilter');
  if (!bar) return;
  const t = I18N[lang];
  const chips = [
    { key: 'All', icon: '🌾', label: t.filter_all },
    { key: 'Enset', icon: '🌱', label: t.crop_enset },
    { key: 'Coffee', icon: '☕', label: t.crop_coffee },
    { key: 'Maize', icon: '🌽', label: t.crop_maize }
  ];
  bar.innerHTML = chips.map((c) =>
    '<button type="button" class="crop-chip' + (dashCropFilter === c.key ? ' active' : '') + '"' +
    ' data-crop="' + c.key + '" aria-pressed="' + (dashCropFilter === c.key) + '">' +
    c.icon + ' ' + esc(c.label) + '</button>'
  ).join('');
}

/** True when an entity's crop passes the active dashboard filter. */
function cropMatches(crop) {
  if (dashCropFilter === 'All') return true;
  return !!crop && String(crop).toLowerCase().indexOf(dashCropFilter.toLowerCase()) !== -1;
}

/** FIX (unified dashboard): renderDashboard — single disease-list view.
 *  Renders every catalog disease that passes the crop filter as an accordion
 *  card with 6 independently collapsible sub-sections:
 *    🔍 Symptoms, 🛡️ Prevention, 💊 Treatment, 🏠 Traditional Remedies,
 *    📅 Seasonality, 🌿 Resistant Varieties
 *  The card header toggles the whole card; each sub-section has its own
 *  data-action="toggle-sub" toggle so they expand/collapse independently.
 */
function renderDashboard() {
  if (!dashBody) return;
  const t = I18N[lang];
  const classes = (catalog && Array.isArray(catalog.classes)) ? catalog.classes : [];
  const byId = {};
  const dashRows = (dashData && Array.isArray(dashData.diseases)) ? dashData.diseases : [];
  dashRows.forEach((d) => { byId[d.id] = d; });
  const alerts = (dashData && Array.isArray(dashData.threat_alerts)) ? dashData.threat_alerts : [];
  // Risk label: prefer the localized label from the catalog risk object.
  const riskLabel = (risk) => {
    if (!risk) return '';
    return (lang === 'am' && risk.label_am) ? risk.label_am : (risk.label || '');
  };
  // A guard against a blank page when the catalog (primary source) is absent.
  if (classes.length === 0) {
    dashBody.innerHTML = '<div class="dash-card"><p>' +
      (lang === 'am' ? 'የበሽታ ውሂብ አልተጫነም። እባክዎ ገጹን ያድሱ።' : 'Disease data not loaded \u2013 please refresh the page.') +
      '</p></div>';
    dashBody.classList.remove('dash-fade-in');
    void dashBody.offsetWidth;
    dashBody.classList.add('dash-fade-in');
    return;
  }

  const filtered = classes.filter((cls) => cropMatches(cls.crop));
  let html = '<h3 class="dash-section-title">' + esc(t.diseases_title) + ' (' + filtered.length +
    '/' + classes.length + ')</h3>';

  // FIX (disease detail): dashboard is a list of tappable cards (NO accordion).
  // Each card has a thumbnail, name, crop, risk badge, and a chevron ›.
  const riskBadge = (risk) => {
    if (!risk) return '';
    const lvl = risk.level || 'normal';
    const label = (lang === 'am' ? risk.label_am : risk.label) || '';
    return '<span class="risk-badge risk-' + esc(lvl) + '">' + esc(label) + '</span>';
  };
  const img = (cls) => {
    const src = cls.image || 'default-disease.svg';
    return '<img class="dash-thumb" src="' + esc(src) + '" alt="" loading="lazy" onerror="this.onerror=null;this.src=\'default-disease.svg\';">';
  };

  filtered.forEach((cls) => {
    const name = (lang === 'am' && cls.name_am) ? cls.name_am : cls.name;
    const crop = (lang === 'am' && cls.crop_am) ? cls.crop_am : cls.crop;
    const risk = cls.risk || {};
    html += '<button type="button" class="dash-card dash-clickable" data-action="open-detail" data-key="' + esc(cls.class_id) + '">' +
      img(cls) +
      '<span class="dash-card-body">' +
        '<span class="dash-card-name">' + esc(name) + '</span>' +
        '<span class="dash-card-sub">' + esc(crop) + ' · <em>' + esc(cls.pathogen || '') + '</em></span>' +
        riskBadge(risk) +
      '</span>' +
      '<span class="dash-card-arrow" aria-hidden="true">›</span>' +
    '</button>';
  });

  // Fragmentation / fade-in for filter/language changes
  dashBody.innerHTML = html;
  dashBody.classList.remove('dash-fade-in');
  void dashBody.offsetWidth; // restart CSS animation
  dashBody.classList.add('dash-fade-in');
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
        li.className = 'history-item history-clickable';
        // FIX (disease detail): a history record is a tappable card that opens the
        // same detail view with scan metadata.
        li.addEventListener('click', () => {
          if (s.inconclusive) return; // inconclusive scans have no disease to open
          openDiseaseDetail(s.classId, true, s);
        });

        // Add a trailing arrow to signal tappability (unless inconclusive)
        const arrow = document.createElement('span');
        arrow.className = 'history-arrow';
        arrow.setAttribute('aria-hidden', 'true');
        arrow.textContent = s.inconclusive ? '' : '›';

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
        li.appendChild(arrow);

        historyList.appendChild(li);
      });
  }

}

function closeHistory() {
  // Clear the #history hash so it doesn't re-open on the next page load.
  if (window.location.hash === '#history') {
    // try/catch: replaceState can throw a SecurityError on file:// URLs.
    try {
      history.replaceState(null, '', window.location.pathname + window.location.search);
    } catch (e) { /* ignore */ }
  }
}

async function clearHistory() {
  await clearAllScans();
  // Re-render immediately so the user sees the emptied list (audit fix).
  await openHistory();
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
