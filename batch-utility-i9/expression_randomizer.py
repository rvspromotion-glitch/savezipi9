"""
ExpressionRandomizerBatch
=========================
Injects a randomly selected facial expression description into each prompt
in a batch.  Each filter option is an individual toggle so you can click
exactly which moods, mouth types, and energy levels are allowed before
sampling.

The same seed produces the same per-image selection sequence across runs.
"""

import logging
import random

logger = logging.getLogger("ExpressionRandomizerBatch")

# ---------------------------------------------------------------------------
# Expression library
# ---------------------------------------------------------------------------

_EXPRESSIONS = [
    {"title":"Genuine laugh","desc":"Open mouth laugh, head slightly back, eyes nearly closed from cheek lift, fully uninhibited","mood":"joyful","mouth":"open","energy":"high","hands":False,"source":"real"},
    {"title":"Hype shout","desc":"Mouth wide open in a raw explosive shout, one eye squinting from force, playful energy underneath","mood":"intense","mouth":"open","energy":"high","hands":False,"source":"real"},
    {"title":"Baddie unbothered","desc":"Lips slightly pouted, jaw relaxed, eyes hidden or half-lidded, peace sign held up close to face at chin/cheek level","mood":"confident","mouth":"pout","energy":"low","hands":True,"hand_position":"peace sign at chin/cheek level","source":"real"},
    {"title":"Warm confident smile","desc":"Soft closed-cheek smile, eyes slightly creased, relaxed and approachable","mood":"warm","mouth":"smile","energy":"medium","hands":False,"source":"real"},
    {"title":"Soft distracted smile","desc":"Slight lip upturn while gazing off to the side, gentle and natural, not performing for camera","mood":"soft","mouth":"smile","energy":"low","hands":False,"source":"real"},
    {"title":"Neutral direct gaze","desc":"Lips closed and relaxed, eyes steady into camera, quietly commanding, no warmth performed","mood":"neutral","mouth":"closed","energy":"low","hands":False,"source":"real"},
    {"title":"Cute wink","desc":"One eye closed in deliberate wink, wide open smile, flirty and self-aware","mood":"flirty","mouth":"smile","energy":"medium","hands":False,"source":"real"},
    {"title":"Coy smirk","desc":"Small asymmetric smile, one eye doing more work than the other, knowing and calculated sweetness","mood":"flirty","mouth":"smile","energy":"low","hands":False,"source":"real"},
    {"title":"Sultry squint pout","desc":"One eye squinting more than the other, lips pushed forward in a soft pout, intense direct gaze into camera","mood":"intense","mouth":"pout","energy":"medium","hands":False,"source":"real"},
    {"title":"Tongue out playful","desc":"Mouth wide open, tongue out, one eye squinting, high energy and cheeky, self-aware","mood":"playful","mouth":"tongue_out","energy":"high","hands":False,"source":"real"},
    {"title":"Cute pout with finger kiss","desc":"Lips pushed into a soft pout, one eye slightly more closed, fingers pinched together pointing upward held at chin/lip level in a delicate mwah gesture","mood":"flirty","mouth":"pout","energy":"low","hands":True,"hand_position":"fingers pinched at chin/lip level","source":"real"},
    {"title":"Doe-eyed resting on hand","desc":"Wide open eyes with soft vacant stare, head resting sideways into open palm at cheek level, lips slightly parted and relaxed, dreamy and gentle","mood":"soft","mouth":"parted","energy":"low","hands":True,"hand_position":"open palm cradling cheek","source":"real"},
    {"title":"Double peace sign grin","desc":"Wide genuine smile, both hands raised with peace signs framing the face at eye level on either side, eyes squinting from the smile","mood":"joyful","mouth":"smile","energy":"high","hands":True,"hand_position":"both hands framing face at eye level","source":"real"},
    {"title":"Laughing behind hand","desc":"Caught mid-laugh, one hand raised covering most of the face, smile breaking through underneath, shy unguarded energy","mood":"playful","mouth":"smile","energy":"medium","hands":True,"hand_position":"hand covering face","source":"real"},
    {"title":"Tongue out eyes closed","desc":"Eyes fully closed, tongue extended downward, lips parted wide, nose slightly scrunched, playful and cheeky","mood":"playful","mouth":"tongue_out","energy":"medium","hands":False,"source":"real"},
    {"title":"Open mouth joy shout","desc":"Mouth wide open mid-shout or singing along, eyes squinted, head slightly back, nose wrinkling from intensity, warm and uninhibited","mood":"joyful","mouth":"open","energy":"high","hands":False,"source":"real"},
    {"title":"Both hands prayer-cupped at cheek","desc":"Soft neutral expression, both hands brought together resting against the side of the face fingers pointing upward, gentle and demure","mood":"soft","mouth":"closed","energy":"low","hands":True,"hand_position":"both hands prayer-cupped at cheek","source":"real"},
    {"title":"One arm raised celebration","desc":"Big open smile, one arm fully extended straight up above the head, body loose and open, pure celebratory energy","mood":"joyful","mouth":"smile","energy":"high","hands":True,"hand_position":"one arm fully raised above head","source":"real"},
    {"title":"Skeptical squint smirk","desc":"One eye squinting more than the other, lips pressed into slight asymmetric smirk, nose slightly scrunched, unimpressed or suspicious attitude","mood":"attitude","mouth":"smile","energy":"low","hands":False,"source":"real"},
    {"title":"Soft smile with cheek peace sign","desc":"Gentle closed-mouth smile, cheeks slightly pushed up, one hand resting against cheek and the other holding a peace sign at face level beside it","mood":"warm","mouth":"smile","energy":"low","hands":True,"hand_position":"one hand on cheek, one peace sign at face level","source":"real"},
    {"title":"Eyes down lip bite","desc":"Gaze directed downward, lower lip subtly caught between the teeth, introspective and slightly tense energy","mood":"soft","mouth":"bite","energy":"low","hands":False,"source":"real"},
    {"title":"Both hands in hair looking away","desc":"Neutral to slightly serious expression, both hands raised holding hair above the head, gaze directed off to the side, editorial and confident","mood":"confident","mouth":"closed","energy":"medium","hands":True,"hand_position":"both hands raised holding hair above head","source":"real"},
    {"title":"Mirror selfie pout","desc":"Lips pushed into a deliberate pout, gaze directed at phone in hand, body angled sideways. Mirror selfie.","mood":"attitude","mouth":"pout","energy":"low","hands":True,"hand_position":"one hand holding phone toward mirror","source":"real"},
    {"title":"Mirror selfie tongue out","desc":"Tongue extended out, eyes looking at phone in hand, cheeky and casual. Mirror selfie.","mood":"playful","mouth":"tongue_out","energy":"medium","hands":True,"hand_position":"one hand holding phone toward mirror","source":"real"},
    {"title":"Surprised lips parted","desc":"Lips parted in a soft O shape, eyes wide, brows slightly raised, caught-off-guard or mock surprise. Mirror selfie.","mood":"neutral","mouth":"parted","energy":"medium","hands":True,"hand_position":"one hand holding phone toward mirror","source":"real"},
    {"title":"Sideways smirk at phone","desc":"Small asymmetric smirk, gaze directed down at phone in hand, nonchalant and slightly amused. Mirror selfie.","mood":"attitude","mouth":"smile","energy":"low","hands":True,"hand_position":"one hand holding phone toward mirror","source":"real"},
    {"title":"Puffed cheeks pout","desc":"Cheeks puffed with air, lips pushed forward into a pout simultaneously, silly and playful. Mirror selfie.","mood":"playful","mouth":"pout","energy":"medium","hands":True,"hand_position":"one hand holding phone toward mirror","source":"real"},
    {"title":"Chin tilt smirk","desc":"Chin slightly lifted upward, lips curled into a light one-sided smirk, gaze looking slightly downward into camera with quiet superiority","mood":"confident","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Biting finger tip","desc":"One finger brought up to the mouth with the tip gently between the teeth, eyes wide and soft, lips slightly parted around it, coy and teasing","mood":"flirty","mouth":"bite","energy":"low","hands":True,"hand_position":"one finger raised to lips","source":"ai"},
    {"title":"Over shoulder glance","desc":"Head turned looking back over one shoulder, lips closed in a subtle soft smile, eyes catching the camera mid-turn, effortlessly alluring","mood":"flirty","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Closed eye serene smile","desc":"Eyes fully closed, soft genuine smile, head tilted slightly, expression of pure contentment and calm","mood":"soft","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Hand framing jaw","desc":"One hand brought up with fingers loosely curled under the jaw, tilting the face slightly upward, lips relaxed and soft, editorial and composed","mood":"neutral","mouth":"closed","energy":"low","hands":True,"hand_position":"fingers loosely framing jaw from below","source":"ai"},
    {"title":"Teeth bared fierce","desc":"Both rows of teeth fully visible in a wide intense grin, eyes sharp and narrowed, high energy and commanding, not warm — fierce","mood":"intense","mouth":"open","energy":"high","hands":False,"source":"ai"},
    {"title":"Nose scrunch grin","desc":"Full wide smile causing the nose to scrunch upward, eyes crinkled, cheeks high and full, authentically cute and unguarded","mood":"joyful","mouth":"smile","energy":"medium","hands":False,"source":"ai"},
    {"title":"Pouty lip pull down","desc":"Bottom lip pulled down slightly with fingertip, exposing bottom teeth edge, eyes direct and teasing into camera","mood":"flirty","mouth":"pout","energy":"medium","hands":True,"hand_position":"one fingertip pulling lightly at lower lip","source":"ai"},
    {"title":"Thinking face look up","desc":"Eyes directed upward and to one side, lips pressed lightly together or slightly pursed, one finger raised to the temple, mock-contemplative and playful","mood":"playful","mouth":"closed","energy":"low","hands":True,"hand_position":"one finger raised to temple","source":"ai"},
    {"title":"Whispering lean in","desc":"Head tilted slightly forward, one hand raised to the side of the mouth as if shielding a whisper, eyes wide and conspiratorial","mood":"playful","mouth":"parted","energy":"medium","hands":True,"hand_position":"one hand cupped at side of mouth","source":"ai"},
    {"title":"Blowing a kiss","desc":"Lips pursed forward mid-blow, one hand raised flat in front of the mouth fingers together as if sending the kiss forward, eyes soft and warm","mood":"flirty","mouth":"pout","energy":"medium","hands":True,"hand_position":"one hand raised flat in front of mouth","source":"ai"},
    {"title":"Dead stare smirk","desc":"Completely flat expressionless eyes, no brow movement, small barely-there smirk at one corner of the mouth, deeply dry and deadpan","mood":"attitude","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Shy look away cover mouth","desc":"Gaze directed off to the side or downward, one hand raised loosely covering the lower half of the face at mouth level, smile hidden behind it","mood":"soft","mouth":"smile","energy":"low","hands":True,"hand_position":"one hand loosely covering mouth","source":"ai"},
    {"title":"Double hand frame face","desc":"Both hands raised on either side of the face with palms forward and fingers spread, framing the face, wide eyes and open expression","mood":"playful","mouth":"parted","energy":"high","hands":True,"hand_position":"both palms framing face at either side","source":"ai"},
    {"title":"Soft upper lip bite","desc":"Upper lip caught lightly between the teeth rather than lower, gaze steady and direct into camera, subtle and intentional","mood":"flirty","mouth":"bite","energy":"low","hands":False,"source":"ai"},
    {"title":"Eyes closed tongue out lean","desc":"Eyes closed, tongue extended to one side of the mouth rather than straight out, head tilted slightly in the same direction, loose and goofy","mood":"playful","mouth":"tongue_out","energy":"medium","hands":False,"source":"ai"},
    {"title":"Forehead rest on hands","desc":"Both hands stacked or laced together with forehead resting down onto them, eyes looking up at camera from below, soft and intimate","mood":"soft","mouth":"closed","energy":"low","hands":True,"hand_position":"both hands stacked under forehead","source":"ai"},
    {"title":"Smug side eye","desc":"Head facing mostly forward but eyes cut sharply to one side, one brow slightly raised, lips closed in a barely suppressed smirk","mood":"attitude","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Open mouth shock grin","desc":"Mouth dropped wide open in exaggerated mock shock, eyes wide and brows raised high, caught mid-reaction, fully performative","mood":"playful","mouth":"open","energy":"high","hands":False,"source":"ai"},
    {"title":"Hand on chest sincere","desc":"One hand placed flat against the chest, soft open smile, eyes warm and direct, genuine and heartfelt energy","mood":"warm","mouth":"smile","energy":"medium","hands":True,"hand_position":"one hand flat on chest","source":"ai"},
    {"title":"Mirror selfie eyes closed pout","desc":"Eyes fully closed, lips pushed into a deliberate pout, free hand resting loosely at side. Mirror selfie.","mood":"flirty","mouth":"pout","energy":"low","hands":True,"hand_position":"one hand holding phone toward mirror","source":"ai"},
    {"title":"Two finger gun point","desc":"One hand raised with index and middle finger extended pointing forward like a gun toward camera, other hand on hip, lips in a confident smirk","mood":"confident","mouth":"smile","energy":"medium","hands":True,"hand_position":"one hand finger-gun pointing toward camera at chest level","source":"ai"},
    {"title":"Sultry tongue corner","desc":"Tongue tip pushed into the corner of the mouth, lips barely parted, eyes half-lidded and direct, slow and deliberate","mood":"intense","mouth":"tongue_out","energy":"low","hands":False,"source":"ai"},
    {"title":"Full beam eyes crinkled","desc":"Widest possible smile, cheeks pushed so high the eyes become thin creases, head slightly tilted, completely unguarded joy","mood":"joyful","mouth":"smile","energy":"high","hands":False,"source":"ai"},
    {"title":"Laughing mid-clap","desc":"Mouth open in a genuine laugh, both hands clapping together in front of the chest at mid-torso level, body leaning slightly forward","mood":"joyful","mouth":"open","energy":"high","hands":True,"hand_position":"both hands clapping at chest level","source":"ai"},
    {"title":"Teeth smile hands on cheeks","desc":"Big open-teeth smile, both hands pressed flat against the cheeks pushing them slightly upward, eyes bright and wide","mood":"joyful","mouth":"smile","energy":"high","hands":True,"hand_position":"both hands pressed to cheeks","source":"ai"},
    {"title":"Jumping arms out","desc":"Mouth open in a shout of joy, both arms flung out wide to the sides at shoulder height, body mid-jump energy","mood":"joyful","mouth":"open","energy":"high","hands":True,"hand_position":"both arms spread wide at shoulder level","source":"ai"},
    {"title":"Giggling eyes squeezed","desc":"Eyes squeezed fully shut from laughing, mouth open in a wide grin, shoulders slightly raised","mood":"joyful","mouth":"open","energy":"medium","hands":False,"source":"ai"},
    {"title":"Cheerful wink point","desc":"One eye closed in a wink, wide smile, one hand raised with index finger pointing outward at camera level","mood":"joyful","mouth":"smile","energy":"medium","hands":True,"hand_position":"one finger pointing forward at camera level","source":"ai"},
    {"title":"Smile hands clasped overhead","desc":"Beaming smile, both arms raised and hands clasped together above the head in a victory pose","mood":"joyful","mouth":"smile","energy":"high","hands":True,"hand_position":"both hands clasped above head","source":"ai"},
    {"title":"Laughing look sideways","desc":"Open mouth laugh, head turned to one side mid-laugh as if reacting to something off-camera, eyes nearly closed","mood":"joyful","mouth":"open","energy":"high","hands":False,"source":"ai"},
    {"title":"Grin tongue between teeth","desc":"Wide smile with tongue lightly resting between the upper and lower front teeth, playful and sweet","mood":"joyful","mouth":"smile","energy":"medium","hands":False,"source":"ai"},
    {"title":"Hard stare closed jaw","desc":"Jaw set firm, lips pressed tightly together in a straight line, eyes unblinking and locked directly at camera, heavy and still","mood":"intense","mouth":"closed","energy":"medium","hands":False,"source":"ai"},
    {"title":"Rage open mouth","desc":"Mouth opened wide in a raw scream, brows pulled hard downward, tension visible across the whole face, full emotional release","mood":"intense","mouth":"open","energy":"high","hands":False,"source":"ai"},
    {"title":"Intense whisper lips","desc":"Lips barely parted, teeth slightly visible, eyes narrowed and forward, expression of quiet menace or urgency","mood":"intense","mouth":"parted","energy":"medium","hands":False,"source":"ai"},
    {"title":"Gripping own collar","desc":"One or both hands gripping the neckline of clothing at chest level, jaw tight, eyes fierce and forward","mood":"intense","mouth":"closed","energy":"high","hands":True,"hand_position":"hands gripping collar at chest level","source":"ai"},
    {"title":"Clenched smile tension","desc":"Smile held with visible tension, teeth slightly clenched, eyes burning and direct — controlled intensity leaking through a grin","mood":"intense","mouth":"smile","energy":"medium","hands":False,"source":"ai"},
    {"title":"Exhale eyes closed","desc":"Eyes fully closed, lips slightly parted as if exhaling slowly, head tilted back slightly, releasing pressure","mood":"intense","mouth":"parted","energy":"low","hands":False,"source":"ai"},
    {"title":"Fierce brow pout","desc":"Brows pulled sharply downward in a hard furrow, lips pushed into a strong pout, eyes locked forward — defiant and brooding","mood":"intense","mouth":"pout","energy":"medium","hands":False,"source":"ai"},
    {"title":"Two hands grabbing own face","desc":"Both hands pressed against the sides of the face, mouth open in exaggerated stress or disbelief, elbows out","mood":"intense","mouth":"open","energy":"high","hands":True,"hand_position":"both hands pressed to sides of face","source":"ai"},
    {"title":"Arms crossed soft smirk","desc":"Both arms crossed loosely at chest level, lips in a light one-sided smirk, gaze steady and unhurried","mood":"confident","mouth":"smile","energy":"low","hands":True,"hand_position":"arms crossed at chest","source":"ai"},
    {"title":"Hand on hip side smile","desc":"One hand resting on the hip, body angled slightly sideways, soft open smile, relaxed and self-assured","mood":"confident","mouth":"smile","energy":"medium","hands":True,"hand_position":"one hand on hip","source":"ai"},
    {"title":"Chin up neutral","desc":"Chin lifted slightly, lips closed and neutral, gaze angled slightly downward at camera, effortlessly commanding","mood":"confident","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Direct smile no teeth","desc":"Closed-mouth smile, completely symmetrical, eyes forward and calm — warm confidence with no effort performed","mood":"confident","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Shoulder lean smirk","desc":"One shoulder raised slightly, head tilted toward it, lips in a quiet smirk, gaze forward — effortlessly cool","mood":"confident","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Slow blink soft smile","desc":"Eyes half-closed in a deliberate slow blink, soft smile at the lips, deeply calm and unbothered","mood":"confident","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Standing tall open smile","desc":"Posture fully upright, wide open genuine smile, chin level, both hands relaxed at sides — grounded and radiant","mood":"confident","mouth":"smile","energy":"medium","hands":False,"source":"ai"},
    {"title":"One brow raised smirk","desc":"One brow arched high, the other neutral, lips in a restrained smirk — skeptical confidence, silently judging","mood":"confident","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Thumbs up grin","desc":"One or both thumbs raised at chest level, wide open grin, eyes bright — classic unironic confidence","mood":"confident","mouth":"smile","energy":"medium","hands":True,"hand_position":"thumbs raised at chest level","source":"ai"},
    {"title":"Soft laugh head tilt","desc":"Small laugh escaping, head tilted gently to one side, eyes warm and slightly crinkled, completely at ease","mood":"warm","mouth":"open","energy":"low","hands":False,"source":"ai"},
    {"title":"Both hands offered forward","desc":"Both hands extended slightly forward with palms open and upward at waist level, soft open smile, inviting and open","mood":"warm","mouth":"smile","energy":"low","hands":True,"hand_position":"both palms open extended forward at waist level","source":"ai"},
    {"title":"Slow warm smile building","desc":"Smile starting small at the corners and spreading, eyes softening as it grows — caught mid-expression, genuine","mood":"warm","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Hug self content","desc":"Both arms wrapped around own torso in a self-hug, chin slightly down, soft smile, cozy and content","mood":"warm","mouth":"smile","energy":"low","hands":True,"hand_position":"both arms wrapped around own torso","source":"ai"},
    {"title":"Grateful smile eyes down","desc":"Eyes cast slightly downward, full warm smile, expression of sincere gratitude or quiet happiness","mood":"warm","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Cheek rest warm gaze","desc":"Head resting sideways onto one raised shoulder, lips in a gentle smile, eyes soft and direct — cozy and inviting","mood":"warm","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Open arms welcoming","desc":"Both arms spread open wide at shoulder height as if offering a hug, big warm smile, eyes bright","mood":"warm","mouth":"smile","energy":"high","hands":True,"hand_position":"both arms spread open at shoulder height","source":"ai"},
    {"title":"Nose touch gentle","desc":"One finger resting lightly on the tip of the nose, soft closed smile, warm and playfully endearing","mood":"warm","mouth":"smile","energy":"low","hands":True,"hand_position":"one finger touching tip of nose","source":"ai"},
    {"title":"Dreamy upward gaze","desc":"Eyes directed softly upward, lips relaxed and barely parted, lost in thought — quiet and introspective","mood":"soft","mouth":"parted","energy":"low","hands":False,"source":"ai"},
    {"title":"Fingers to lips thinking","desc":"Two or three fingers resting lightly against closed lips, gaze directed slightly downward or to the side, thoughtful and still","mood":"soft","mouth":"closed","energy":"low","hands":True,"hand_position":"fingers resting at lips","source":"ai"},
    {"title":"Eyes barely open soft","desc":"Eyes open only a sliver, heavy-lidded and slow, lips relaxed and closed — drowsy or deeply at peace","mood":"soft","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Head bow gentle smile","desc":"Head tilted slightly downward, small gentle smile at the lips, gaze angled up toward camera through lowered lashes","mood":"soft","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Knees to chest curl","desc":"Body curled with knees drawn up, arms loosely wrapped around them, chin resting on knees, expression soft and closed-off in a cozy way","mood":"soft","mouth":"closed","energy":"low","hands":True,"hand_position":"arms wrapped around pulled-up knees","source":"ai"},
    {"title":"Eyes closed deep breath","desc":"Eyes fully closed, lips closed and relaxed, expression of calm deliberate breath — serene and centred","mood":"soft","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Fingertip on cheek","desc":"One fingertip resting lightly against the cheek, head tilted slightly toward it, soft neutral expression","mood":"soft","mouth":"closed","energy":"low","hands":True,"hand_position":"one fingertip resting on cheek","source":"ai"},
    {"title":"Wistful side gaze","desc":"Eyes drifting to one side with a faint sadness or longing, lips closed and slightly downturned at the corners","mood":"soft","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Soft smile palms together","desc":"Both palms pressed together in front of the chest like a soft prayer, gentle closed-mouth smile, peaceful","mood":"soft","mouth":"smile","energy":"low","hands":True,"hand_position":"both palms pressed together at chest level","source":"ai"},
    {"title":"Blank forward stare","desc":"Eyes open and forward, no muscle engagement anywhere on the face, lips closed and completely relaxed — pure neutral","mood":"neutral","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Slight head tilt neutral","desc":"Head tilted a few degrees to one side, expression entirely neutral, eyes forward and calm — observing without reacting","mood":"neutral","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Arms at sides open","desc":"Both arms hanging relaxed at the sides, posture neutral and upright, lips closed, face soft and uneventful","mood":"neutral","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Hand resting on chin","desc":"One hand raised with the chin resting lightly on it, elbow implied below frame, gaze forward and calm — contemplative but not emotive","mood":"neutral","mouth":"closed","energy":"low","hands":True,"hand_position":"chin resting on one hand","source":"ai"},
    {"title":"Profile neutral","desc":"Face turned fully to one side showing the profile, expression neutral and composed, lips closed","mood":"neutral","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Mouth slightly open neutral","desc":"Lips just barely separated, no expression pushed into the face, eyes forward and soft — caught at rest","mood":"neutral","mouth":"parted","energy":"low","hands":False,"source":"ai"},
    {"title":"Looking down neutral","desc":"Gaze directed downward, face relaxed and expressionless, neither happy nor sad — simply still","mood":"neutral","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Lip corner pull smile","desc":"One corner of the lip pulled back slowly into a knowing half-smile, eyes locked on camera with quiet intent","mood":"flirty","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Slow wink no smile","desc":"One eye closing in a slow deliberate wink, lips completely neutral — no smile, all intention","mood":"flirty","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Hair tuck behind ear","desc":"One hand raised to gently tuck hair behind the ear, soft closed smile, gaze slightly downward then up","mood":"flirty","mouth":"smile","energy":"low","hands":True,"hand_position":"one hand raised to ear level tucking hair","source":"ai"},
    {"title":"Finger trace jaw","desc":"One finger trailing lightly along the jawline from chin toward the ear, lips parted softly, eyes forward and direct","mood":"flirty","mouth":"parted","energy":"low","hands":True,"hand_position":"one finger tracing along jawline","source":"ai"},
    {"title":"Teeth smile eyes half closed","desc":"Wide smile showing teeth, eyes lowered to half-closed heavy-lidded — warm but deliberately seductive","mood":"flirty","mouth":"smile","energy":"medium","hands":False,"source":"ai"},
    {"title":"Head tilt pout","desc":"Head tilted to one side, lips pushed into a soft pout, eyes wide and forward — sweet and deliberate","mood":"flirty","mouth":"pout","energy":"low","hands":False,"source":"ai"},
    {"title":"Collar pull tease","desc":"One hand pulling lightly at the neckline of clothing downward at chest level, lips in a coy smile, eyes up at camera","mood":"flirty","mouth":"smile","energy":"medium","hands":True,"hand_position":"one hand pulling at neckline at chest level","source":"ai"},
    {"title":"Wrist to chin tilt","desc":"Back of one wrist resting under the chin, tilting the face slightly upward, lips soft and parted, eyes dreamy and forward","mood":"flirty","mouth":"parted","energy":"low","hands":True,"hand_position":"back of wrist resting under chin","source":"ai"},
    {"title":"Tongue to upper lip","desc":"Tip of tongue resting against or tracing the upper lip, eyes forward and calm, slow and intentional","mood":"flirty","mouth":"tongue_out","energy":"low","hands":False,"source":"ai"},
    {"title":"Finger under chin look up","desc":"One finger placed under the chin pushing it slightly upward, eyes gazing up toward camera, lips softly parted","mood":"flirty","mouth":"parted","energy":"low","hands":True,"hand_position":"one finger under chin","source":"ai"},
    {"title":"Raspberry blow","desc":"Lips pushed forward and vibrating in a raspberry, cheeks puffed slightly, eyes scrunched from effort — fully silly","mood":"playful","mouth":"open","energy":"medium","hands":False,"source":"ai"},
    {"title":"Crossed eyes grin","desc":"Eyes deliberately crossed, wide grin, completely goofy and self-aware — performed silliness","mood":"playful","mouth":"smile","energy":"medium","hands":False,"source":"ai"},
    {"title":"Tongue out wink point","desc":"One eye closed in a wink, tongue out to one side, one hand raised with index finger pointing at camera at chest level","mood":"playful","mouth":"tongue_out","energy":"high","hands":True,"hand_position":"one finger pointing at camera at chest level","source":"ai"},
    {"title":"Bunny ears behind own head","desc":"Both hands raised behind the head with index and middle fingers up making bunny ears, wide grin","mood":"playful","mouth":"smile","energy":"medium","hands":True,"hand_position":"both hands making bunny ears behind head","source":"ai"},
    {"title":"Exaggerated shocked pout","desc":"Eyes blown wide, lips pushed into an extreme exaggerated pout — mock offended or overdramatic reaction","mood":"playful","mouth":"pout","energy":"medium","hands":False,"source":"ai"},
    {"title":"Silly salute grin","desc":"One hand raised flat to the forehead in a salute, wide goofy grin, completely unserious","mood":"playful","mouth":"smile","energy":"medium","hands":True,"hand_position":"one hand flat salute at forehead","source":"ai"},
    {"title":"Peek from behind hands","desc":"Both hands raised covering the face, with eyes peeking out from just above the fingertips, playful hiding expression","mood":"playful","mouth":"closed","energy":"medium","hands":True,"hand_position":"both hands covering face with eyes peeking above","source":"ai"},
    {"title":"Fist pump open mouth","desc":"One fist punched upward above the shoulder in a pump, mouth open in a triumphant shout, high energy","mood":"playful","mouth":"open","energy":"high","hands":True,"hand_position":"one fist raised above shoulder in a pump","source":"ai"},
    {"title":"Cheeks pinched self","desc":"Both hands reaching up to lightly pinch own cheeks between thumb and index finger, eyes wide and soft, silly and endearing","mood":"playful","mouth":"closed","energy":"low","hands":True,"hand_position":"both hands pinching own cheeks","source":"ai"},
    {"title":"Air quotes smirk","desc":"Both hands raised with index and middle fingers curling into air quotes at face level, lips in a knowing smirk","mood":"playful","mouth":"smile","energy":"medium","hands":True,"hand_position":"both hands making air quotes at face level","source":"ai"},
    {"title":"Eye roll slight smirk","desc":"Eyes rolled upward with visible effort, small smirk at the corner of the lips — unbothered and done with it","mood":"attitude","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Flat bored stare","desc":"Eyes open, expression completely flat, lips neutral — no irritation, just zero interest, deeply unimpressed","mood":"attitude","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Hand up stop","desc":"One hand raised flat toward camera with palm facing out at face level — assertive stop gesture, expression neutral or firm","mood":"attitude","mouth":"closed","energy":"medium","hands":True,"hand_position":"one palm raised flat toward camera at face level","source":"ai"},
    {"title":"Double arm cross cold stare","desc":"Both arms folded tightly across the chest, eyes direct and unblinking, lips pressed into a flat line — cold and closed off","mood":"attitude","mouth":"closed","energy":"low","hands":True,"hand_position":"arms tightly crossed at chest","source":"ai"},
    {"title":"Jaw drop unimpressed","desc":"Mouth hanging open slightly in a slow unimpressed drop, one brow raised, eyes flat — the look of silent disbelief","mood":"attitude","mouth":"open","energy":"low","hands":False,"source":"ai"},
    {"title":"Dismissive glance away","desc":"Face turned slightly away, eyes cutting to the side dismissively, lips closed and neutral — pointedly uninterested","mood":"attitude","mouth":"closed","energy":"low","hands":False,"source":"ai"},
    {"title":"Lip curl disgust","desc":"One side of the upper lip curling upward in a subtle sneer, eyes slightly narrowed, quietly contemptuous","mood":"attitude","mouth":"smile","energy":"low","hands":False,"source":"ai"},
    {"title":"Finger wag no","desc":"One index finger raised and wagging side to side at face level, lips pressed into a firm line or smirk","mood":"attitude","mouth":"closed","energy":"medium","hands":True,"hand_position":"one finger wagging at face level","source":"ai"},
    {"title":"Slow clap smirk","desc":"Both hands brought together in slow deliberate claps held at chest level, lips in a dry smirk — sarcastic applause energy","mood":"attitude","mouth":"smile","energy":"low","hands":True,"hand_position":"both hands clapping slowly at chest level","source":"ai"},
    {"title":"UwU soft mouth","desc":"Lips curved into a small soft curved smile with the corners pressed gently upward and slightly inward, cheeks rounded and high, expression warm and yielding","mood":"soft","mouth":"smile","energy":"low","hands":False,"source":"kawaii"},
    {"title":"Teary eyes happy smile","desc":"Eyes glistening with unshed tears at the lower lids while the mouth holds a wide trembling happy smile, overwhelmed with positive emotion","mood":"joyful","mouth":"smile","energy":"medium","hands":False,"source":"kawaii"},
    {"title":"Pout with watery eyes","desc":"Lips pushed into a full trembling pout, eyes wide and glistening as if about to cry, brows angled upward at the inner corners — classic pleading expression","mood":"soft","mouth":"pout","energy":"low","hands":False,"source":"kawaii"},
    {"title":"Cat mouth smile","desc":"Mouth curved into a small w-shaped or wavy cat-like smile, cheeks soft and full, expression gentle and endearing","mood":"warm","mouth":"smile","energy":"low","hands":False,"source":"kawaii"},
    {"title":"Both hands up peace wink","desc":"Both hands raised with peace signs on either side of the face, one eye closed in a wink, wide bright smile","mood":"joyful","mouth":"smile","energy":"high","hands":True,"hand_position":"both peace signs raised at either side of face","source":"kawaii"},
    {"title":"Flushed cheeks closed smile","desc":"Cheeks visibly flushed and rounded, lips pressed into a small shy closed smile, eyes slightly downcast — bashful and sweet","mood":"flirty","mouth":"smile","energy":"low","hands":False,"source":"kawaii"},
    {"title":"Surprised O mouth hands to cheeks","desc":"Mouth dropped into a perfect O of surprise, both hands pressed to cheeks, eyes wide — exaggerated kawaii shock","mood":"playful","mouth":"open","energy":"high","hands":True,"hand_position":"both hands pressed to cheeks","source":"kawaii"},
    {"title":"Peeking one eye","desc":"One hand raised covering half the face at nose level, one eye peeking over the top of it, shy and curious expression","mood":"soft","mouth":"closed","energy":"low","hands":True,"hand_position":"one hand covering half the face at nose level","source":"kawaii"},
    {"title":"Tongue out peace sideways","desc":"Tongue out to one side, one eye closed in a wink, one hand raised with a peace sign beside the cheek at face level","mood":"playful","mouth":"tongue_out","energy":"medium","hands":True,"hand_position":"one peace sign raised beside cheek at face level","source":"kawaii"},
    {"title":"Shy finger point cheek","desc":"One finger pointing into the cheek with light pressure, head tilted toward it, small shy smile — innocent and coy","mood":"flirty","mouth":"smile","energy":"low","hands":True,"hand_position":"one finger pressing into cheek","source":"kawaii"},
    {"title":"Hands clasped chin rest","desc":"Both hands clasped together just below the chin with the chin resting on top, eyes wide and forward, soft expectant expression","mood":"soft","mouth":"closed","energy":"low","hands":True,"hand_position":"both hands clasped under chin","source":"kawaii"},
    {"title":"Victory arms smile","desc":"Both arms raised in a V above the head, wide beaming smile, full celebratory energy — classic anime victory pose","mood":"joyful","mouth":"smile","energy":"high","hands":True,"hand_position":"both arms raised in V above head","source":"kawaii"},
    {"title":"Embarrassed cover face","desc":"Both hands raised fully covering the face at eye level, body language suggesting a cringe or laugh, peeking energy implied","mood":"soft","mouth":"closed","energy":"medium","hands":True,"hand_position":"both hands fully covering face","source":"kawaii"},
    {"title":"Determined clenched fist","desc":"One fist raised at chin level, lips pressed into a firm determined line or small fierce smile, eyes intense and focused — resolved","mood":"intense","mouth":"smile","energy":"medium","hands":True,"hand_position":"one fist raised at chin level","source":"kawaii"},
    {"title":"Sparkle eye open smile","desc":"Eyes wide and bright with an animated alert quality, mouth open in a delighted smile, expression of pure wonder and excitement","mood":"joyful","mouth":"open","energy":"high","hands":False,"source":"kawaii"},
    {"title":"Smug anime grin","desc":"Wide grin held with visible smugness, one eye slightly more closed than the other, body language relaxed and self-satisfied","mood":"attitude","mouth":"smile","energy":"medium","hands":False,"source":"kawaii"},
    {"title":"Head tilt curious pout","desc":"Head tilted to one side, lips in a small soft pout, expression quietly curious or mildly confused — endearing and gentle","mood":"soft","mouth":"pout","energy":"low","hands":False,"source":"kawaii"},
    {"title":"Fist to palm resolve","desc":"One fist tapped into the open palm of the other hand at chest level, lips in a small determined smile, eyes bright — aha moment pose","mood":"confident","mouth":"smile","energy":"medium","hands":True,"hand_position":"fist tapped into open palm at chest level","source":"kawaii"},
    {"title":"Shy glance up blushing","desc":"Eyes angled upward toward camera from a lowered face, lips closed in a barely-there smile, expression of bashful shyness","mood":"flirty","mouth":"smile","energy":"low","hands":False,"source":"kawaii"},
    {"title":"Hands on own head panic","desc":"Both hands pressed down on top of the head, mouth open in a wide panicked or overwhelmed expression, high distress energy","mood":"intense","mouth":"open","energy":"high","hands":True,"hand_position":"both hands pressed down on top of head","source":"kawaii"},
    {"title":"Cheek puff stare","desc":"Cheeks puffed with air, eyes wide open and staring directly forward, no other expression — blank and puffed","mood":"neutral","mouth":"pout","energy":"low","hands":False,"source":"kawaii"},
    {"title":"Anime cry laugh","desc":"Mouth open in a laugh, eyes scrunched with exaggerated anime-style tears streaming from the corners — crying from laughter","mood":"joyful","mouth":"open","energy":"high","hands":False,"source":"kawaii"},
    {"title":"Soft eyes finger to lips shh","desc":"One finger raised vertically in front of the lips in a shh gesture, eyes soft and slightly narrowed, small subtle smile","mood":"flirty","mouth":"smile","energy":"low","hands":True,"hand_position":"one finger raised vertically at lips","source":"kawaii"},
    {"title":"Nervous laugh hand behind head","desc":"Open awkward laugh, one hand raised and resting behind the head at the back of the skull, sheepish and self-conscious","mood":"playful","mouth":"open","energy":"medium","hands":True,"hand_position":"one hand resting behind head","source":"kawaii"},
    {"title":"Pouty arms crossed sulk","desc":"Both arms crossed firmly at the chest, lips pushed into a full sulky pout, cheeks puffed, eyes averted — theatrically sulking","mood":"attitude","mouth":"pout","energy":"low","hands":True,"hand_position":"arms crossed at chest","source":"kawaii"},
    {"title":"Peace sign cheek squish","desc":"One peace sign pressed against the cheek squishing it slightly, one eye closed in a wink, soft bright smile","mood":"warm","mouth":"smile","energy":"medium","hands":True,"hand_position":"peace sign pressed against cheek","source":"kawaii"},
    {"title":"Pointing up enlightened","desc":"One index finger pointed straight upward above the shoulder, mouth open in a knowing smile or small O, eyes bright — eureka pose","mood":"confident","mouth":"open","energy":"medium","hands":True,"hand_position":"one index finger pointed upward above shoulder","source":"kawaii"},
    {"title":"Heart hands to camera","desc":"Both hands shaped into a heart using thumbs and index fingers, held out toward camera at chest level, soft warm smile","mood":"warm","mouth":"smile","energy":"medium","hands":True,"hand_position":"heart shape made with both hands at chest level","source":"kawaii"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_mirror(expr: dict) -> bool:
    """Detect mirror selfie entries from description/title since field was removed."""
    text = (expr.get("desc", "") + " " + expr.get("title", "")).lower()
    return "mirror selfie" in text


def _build_allowed(flag_map: dict) -> set | None:
    """
    Turn a {value: bool} map into an allowed-set.
    Returns None (= no filter) if everything is on OR everything is off —
    disabling all is treated as 'no restriction' to avoid an empty pool.
    """
    enabled = {k for k, v in flag_map.items() if v}
    if len(enabled) == 0 or len(enabled) == len(flag_map):
        return None
    return enabled


def _filter_pool(
    allowed_moods,
    allowed_mouths,
    allowed_energies,
    hands: str,
    mirror: str,
    source: str,
) -> list:
    pool = []
    for expr in _EXPRESSIONS:
        if allowed_moods    and expr["mood"]   not in allowed_moods:
            continue
        if allowed_mouths   and expr["mouth"]  not in allowed_mouths:
            continue
        if allowed_energies and expr["energy"] not in allowed_energies:
            continue
        if hands == "with hands" and not expr["hands"]:
            continue
        if hands == "no hands" and expr["hands"]:
            continue
        is_mirror = _is_mirror(expr)
        if mirror == "mirror only" and not is_mirror:
            continue
        if mirror == "no mirror" and is_mirror:
            continue
        if source == "real only"   and expr["source"] != "real":
            continue
        if source == "ai only"     and expr["source"] != "ai":
            continue
        if source == "kawaii only" and expr["source"] != "kawaii":
            continue
        pool.append(expr)
    return pool


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _unwrap(val, default):
    if isinstance(val, list):
        return val[0] if val else default
    return val if val is not None else default


class ExpressionRandomizerBatch:
    """
    Randomly injects an expression description into each prompt in a batch.

    Mood, mouth, and energy filters are individual toggles — uncheck a type
    to exclude it.  If every toggle in a group is unchecked the group filter
    is ignored so you never get an empty pool from a single misconfiguration.

    hands / mirror / source are dropdowns.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": True}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31, "step": 1,
                    "tooltip": "Same seed = same expression sequence every run.",
                }),
            },
            "optional": {
                # ── Mood ──────────────────────────────────────────────────
                "mood_joyful":    ("BOOLEAN", {"default": True}),
                "mood_intense":   ("BOOLEAN", {"default": True}),
                "mood_confident": ("BOOLEAN", {"default": True}),
                "mood_warm":      ("BOOLEAN", {"default": True}),
                "mood_soft":      ("BOOLEAN", {"default": True}),
                "mood_neutral":   ("BOOLEAN", {"default": True}),
                "mood_flirty":    ("BOOLEAN", {"default": True}),
                "mood_playful":   ("BOOLEAN", {"default": True}),
                "mood_attitude":  ("BOOLEAN", {"default": True}),
                # ── Mouth ─────────────────────────────────────────────────
                "mouth_open":       ("BOOLEAN", {"default": True}),
                "mouth_pout":       ("BOOLEAN", {"default": True}),
                "mouth_smile":      ("BOOLEAN", {"default": True}),
                "mouth_closed":     ("BOOLEAN", {"default": True}),
                "mouth_tongue_out": ("BOOLEAN", {"default": True}),
                "mouth_parted":     ("BOOLEAN", {"default": True}),
                "mouth_bite":       ("BOOLEAN", {"default": True}),
                # ── Energy ────────────────────────────────────────────────
                "energy_high":   ("BOOLEAN", {"default": True}),
                "energy_medium": ("BOOLEAN", {"default": True}),
                "energy_low":    ("BOOLEAN", {"default": True}),
                # ── Other filters ─────────────────────────────────────────
                "hands":  (["any", "with hands", "no hands"],            {"default": "any"}),
                "mirror": (["any", "mirror only", "no mirror"],          {"default": "any"}),
                "source": (["all", "real only", "ai only", "kawaii only"], {"default": "all"}),
                # ── Output options ────────────────────────────────────────
                "include_title": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Prepend the expression title before its description.",
                }),
                "separator": ("STRING", {
                    "default": ", ",
                    "tooltip": "Glue inserted between the original prompt and the expression text.",
                }),
            },
        }

    RETURN_TYPES   = ("STRING",)
    RETURN_NAMES   = ("prompts",)
    FUNCTION       = "apply"
    CATEGORY       = "Gemini/Creative"

    INPUT_IS_LIST  = (True,)
    OUTPUT_IS_LIST = (True,)

    def apply(
        self,
        prompts: list,
        seed: int = 0,
        mood_joyful: bool = True,
        mood_intense: bool = True,
        mood_confident: bool = True,
        mood_warm: bool = True,
        mood_soft: bool = True,
        mood_neutral: bool = True,
        mood_flirty: bool = True,
        mood_playful: bool = True,
        mood_attitude: bool = True,
        mouth_open: bool = True,
        mouth_pout: bool = True,
        mouth_smile: bool = True,
        mouth_closed: bool = True,
        mouth_tongue_out: bool = True,
        mouth_parted: bool = True,
        mouth_bite: bool = True,
        energy_high: bool = True,
        energy_medium: bool = True,
        energy_low: bool = True,
        hands: str = "any",
        mirror: str = "any",
        source: str = "all",
        include_title: bool = False,
        separator: str = ", ",
    ) -> tuple:
        seed          = _unwrap(seed, 0)
        include_title = _unwrap(include_title, False)
        separator     = _unwrap(separator, ", ")
        hands         = _unwrap(hands, "any")
        mirror        = _unwrap(mirror, "any")
        source        = _unwrap(source, "all")

        mood_joyful    = _unwrap(mood_joyful, True)
        mood_intense   = _unwrap(mood_intense, True)
        mood_confident = _unwrap(mood_confident, True)
        mood_warm      = _unwrap(mood_warm, True)
        mood_soft      = _unwrap(mood_soft, True)
        mood_neutral   = _unwrap(mood_neutral, True)
        mood_flirty    = _unwrap(mood_flirty, True)
        mood_playful   = _unwrap(mood_playful, True)
        mood_attitude  = _unwrap(mood_attitude, True)

        mouth_open       = _unwrap(mouth_open, True)
        mouth_pout       = _unwrap(mouth_pout, True)
        mouth_smile      = _unwrap(mouth_smile, True)
        mouth_closed     = _unwrap(mouth_closed, True)
        mouth_tongue_out = _unwrap(mouth_tongue_out, True)
        mouth_parted     = _unwrap(mouth_parted, True)
        mouth_bite       = _unwrap(mouth_bite, True)

        energy_high   = _unwrap(energy_high, True)
        energy_medium = _unwrap(energy_medium, True)
        energy_low    = _unwrap(energy_low, True)

        allowed_moods = _build_allowed({
            "joyful": mood_joyful, "intense": mood_intense, "confident": mood_confident,
            "warm": mood_warm, "soft": mood_soft, "neutral": mood_neutral,
            "flirty": mood_flirty, "playful": mood_playful, "attitude": mood_attitude,
        })
        allowed_mouths = _build_allowed({
            "open": mouth_open, "pout": mouth_pout, "smile": mouth_smile,
            "closed": mouth_closed, "tongue_out": mouth_tongue_out,
            "parted": mouth_parted, "bite": mouth_bite,
        })
        allowed_energies = _build_allowed({
            "high": energy_high, "medium": energy_medium, "low": energy_low,
        })

        pool = _filter_pool(allowed_moods, allowed_mouths, allowed_energies, hands, mirror, source)

        if not pool:
            logger.warning(
                "ExpressionRandomizerBatch: no expressions match filters — returning prompts unchanged."
            )
            return (list(prompts),)

        logger.info(
            f"ExpressionRandomizerBatch: {len(pool)} expressions in pool, "
            f"{len(prompts)} prompts, seed={seed}"
        )

        rng = random.Random(seed)
        results = []

        for idx, prompt in enumerate(prompts):
            expr = rng.choice(pool)
            expr_text = f"{expr['title']}: {expr['desc']}" if include_title else expr["desc"]
            combined  = f"{prompt}{separator}{expr_text}"
            results.append(combined)
            logger.debug(f"  [{idx}] {expr['title']} → {combined[:100]}")

        logger.info(f"✓ ExpressionRandomizerBatch: {len(results)} prompts processed")
        return (results,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ExpressionRandomizerBatch": ExpressionRandomizerBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpressionRandomizerBatch": "Expression Randomizer (Batch)",
}
