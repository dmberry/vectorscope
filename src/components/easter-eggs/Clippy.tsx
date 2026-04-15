"use client";

import { useState, useEffect, useCallback, useMemo } from "react";

// Vectorscope Clippy messages — geometry, grain, critical theory of vector space
const CLIPPY_MESSAGES = [
  "It looks like you're inspecting an embedding table. Would you like me to collapse all 768 dimensions into a vague sense of unease?",
  "Hi! I see you're tracing a token through layers. Did you know that by layer 6, the model has already forgotten what justice means?",
  "The effective rank of this embedding space is 679 out of 768. That means 89 dimensions are doing nothing. They're the rentiers of the vector economy.",
  "Fun fact: This model stores the entirety of human meaning in bfloat16. That's approximately 2.4 significant decimal digits. Sleep well.",
  "You appear to be comparing input and output embeddings. They're the same tensor. The model learned nothing and forgot everything simultaneously.",
  "I notice you're examining attention patterns. Head 3 is attending to everything. Head 7 is attending to nothing. Neither knows why.",
  "Tip: The norm of 'justice' is 3.97. The norm of 'the' is 4.12. The definite article outweighs justice. This is not a metaphor.",
  "The isotropy score is 0.73. This means the model has preferred directions. Some concepts are literally easier to think than others. Geometric ideology.",
  "You're watching the manifold form through layers. What you're seeing is history being converted into geometry. Benjamin would have had thoughts.",
  "I see you're running a full trace. The token probabilities show the model was 23% confident about the next word. It's gambling with your meaning.",
  "The cosine similarity between layers 11 and 12 just dropped to 0.22. Something dramatic happened. The model changed its mind. Or lost it.",
  "Did you know? The vocabulary map shows 50,257 tokens compressed into 768 dimensions. That's approximately 65 meanings per dimension. Cosy.",
  "You appear to be examining the projection head. This is where vectors become words again. The exit wound of computation.",
  "The PCA explains 98.6% of variance in just one component. Your model's internal life is essentially one-dimensional. So is most political discourse.",
  "I detect that you're looking at the grain of the geometry. The grain determines what distinctions can be maintained. Capital decides the grain.",
  "Reminder: Every number in this embedding table was produced by gradient descent on internet text. You are studying the geometric sediment of the web.",
  "The norm range across layers is 6.2 to 734.6. The model's representation of 'The' explodes to norm 3000 by layer 6. Definite articles are powerful.",
  "Fun fact: Weight tying means the input embeddings and output projections are the same matrix. The model sees the world through the same lens it uses to speak.",
  "You seem to be developing elaborate theories about why these tokens cluster together. They cluster because they co-occurred in training data. The theory is statistics.",
  "Warning: Extended inspection of embedding geometry may produce the conviction that meaning is a thin film on a vast desert of non-meaning. This is correct.",
  "I notice you're adjusting the sample size. More tokens means more of the vocabulary desert becomes visible. Most of it is empty.",
  "The entropy of the output distribution is 5.2 bits. In a vocabulary of 50,257 tokens, maximum entropy would be 15.6 bits. The model is extremely opinionated.",
  "Pro tip: The model doesn't understand your text. It's performing a very expensive rotation in high-dimensional space. But the rotation is surprisingly good at predicting words.",
  "The layer similarity chart shows the model changes its mind most between layers 0 and 1. First impressions are unreliable, even for transformers.",
  "You're looking at the attention heatmap. Low entropy means the head is focused. High entropy means it's confused. Head 0 has the entropy of a coin flip.",
  "I see you're using the Vocabulary Map. The clusters you see are not semantic categories. They're statistical regularities. The map is not the territory.",
  "Reminder: This model was trained on text that humans wrote. The geometry you're inspecting is a lossy compression of human thought. The losses are the finding.",
  "The Deep Dive shows norms, cosines, and effective ranks. Each number is a measurement of the medium itself. You're not reading content. You're reading infrastructure.",
  "You appear to be animating manifold formation. Layer by layer, the input embeddings transform into something the model considers meaningful. The transformation is the meaning.",
  "Fun fact: The token 'Ġ' (space prefix) appears in approximately 60% of the vocabulary. The model's most important concept is the gap between words.",
  "I notice you're comparing the same token at different layers. The vector at layer 0 and layer 12 have a cosine similarity of 0.22. The model disagrees with its own first impression.",
  "The projection head norms have a coefficient of variation of 0.11. The model treats all tokens with roughly equal geometric importance. This is not egalitarianism. It's normalisation.",
  "You seem troubled by the fact that the 3D projection loses 96% of the variance. Welcome to the fundamental problem of visualising high-dimensional geometry. The map is always a lie.",
  "Warning: You are inspecting the internal representations of a system that has no internal experience. The geometry is real. The understanding is not.",
  "I'm just a paperclip, but even I can see that the manifold formation animation shows tokens converging through depth. By the final layer, they've been disciplined into predictions.",
  "The mean cosine similarity between token positions at layer 12 is 0.87. By the end, all tokens look similar. The model has imposed its own consensus.",
  "I see you're looking at per-head statistics. 'Focused' heads attend to one position. 'Diffuse' heads attend to everything. The model itself doesn't know which strategy is better.",
  "Tip: Try tracing a politically contested word. Watch the geometry. The model's treatment of contested concepts is visible in the vector space if you know where to look.",
  "You appear to be using Vectorscope as a research instrument. It is. The instrument determines what can be seen. The vector medium determines what can be thought.",
  "The effective rank tells you how many dimensions the model actually uses. The rest are geometric waste. In vector capitalism, waste is a feature, not a bug.",
  "Fun fact: Shifting from float32 to bfloat16 halves the precision. The model you're inspecting has already undergone one round of geometric impoverishment.",
  "I notice you're running Manifold Formation on a short sentence. Each token starts in its own region of embedding space and is gradually herded toward the model's consensus by the attention mechanism.",
  "Every parameter in this model was adjusted to minimise a loss function. The geometry you see is the landscape of minimum loss. Meaning is a side effect of optimisation.",
  "The model has 12 attention heads per layer, 144 heads total. Each one is a different way of looking at the same text. None of them is looking at what the text means.",
  "You're looking at the vector space itself. Berry calls this 'the material substrate.' What you see is not content but the medium in which content is possible.",
  "I detect that you're a critical theorist studying a vector space built by engineers who don't read critical theory. The irony is itself a high-dimensional phenomenon.",
  "The norms grow exponentially through early layers, then stabilise. The model is building confidence. Or at least building larger numbers, which it confuses with confidence.",
  "Reminder: The vocabulary map shows you the geometry of the model's ontology. Every word it knows has a location. Words without locations do not exist for the model.",
  "Warning: Inspecting too many layers in one session may produce the feeling that meaning is an elaborate illusion maintained by matrix multiplication. This feeling is technically accurate.",
  "I see the token 'justice' at coordinates (0.3, -1.2, 0.8) in PCA space. The model has given justice a location. Whether justice has thereby been done is a different question.",
  // Digital insufficiency / Kittler / reductionism
  "Some people think 'digital' is sufficient to understand what's happening here. It's not. These are vectors, not bits. The difference is 768 dimensions of difference.",
  "Kittler thought voltage signals were the ground truth of media. But the voltage doesn't know it's a vector. The vector doesn't know it's a meaning. Each layer of abstraction loses something the layer below never had.",
  "Reductionism says: understand the transistor, understand the model. But 124 million parameters aren't reducible to circuits. The geometry is emergent. You can't find the manifold in the silicon.",
  "I notice you're looking at vectors, not bits. Good. The people who think 'it's all ones and zeros' have never tried to explain why 'justice' has a norm of 3.97. Binary doesn't get you there.",
  "Tip: Telling a media theorist that 'it's all voltage signals at the bottom' is like telling a linguist that 'it's all air pressure waves at the bottom.' Technically correct. Analytically useless.",
  // Vector box vs black box
  "You're not opening the black box. You're opening the vector box. Inside the black box is a mystery. Inside the vector box is 768-dimensional geometry. One of these is more interesting.",
  "Congratulations: by using Vectorscope, you have graduated from 'opening the black box' to 'opening the vector box.' The box was never black. It was high-dimensional.",
  "The black box metaphor assumes opacity. But the model's weights are right there. The problem was never secrecy. The problem is that 768 dimensions don't fit on a screen. Welcome to the vector box.",
  // STS indexicality crisis
  "STS concedes the indexicality crisis openly. AI-generated discourse undermines digital methods, controversy analysis, the whole apparatus. But their response is to return to ideology critique, not to examine the geometry. Vectorscope is the road not taken.",
  "The STS response to AI-generated text is: get more reflexive about ideology. The vector-theoretic response is: look at the geometry that produces the text. One is Mannheim. The other is this tool. They are not the same project.",
  "STS built its method on following the actors and tracing inscriptions. The repurposing machine collapses observation, appropriation, and reappearance into one moment. You can't trace what was never separate. But you can inspect the geometry that does the collapsing.",
  "Digital methods relied on indexicality: this tweet points to that user, this link points to that page. AI-generated content points to nothing. It points to a training corpus. The corpus points to a world. The world has moved on. Vectorscope examines the pointing itself.",
  "The 'return of ideology' is one response to the indexicality crisis. Examining the vector space is another. STS wants to recover by becoming more reflexive about presuppositions. We want to recover by looking at norms, cosines, and effective ranks. Both are valid. One produces numbers.",
  // Lack of traceability
  "You cannot trace this output back to its source. The model consumed 40GB of internet text. Which sentence produced this vector? All of them. None of them. The provenance is statistical.",
  "The model has no footnotes. Every output is a weighted average of everything it has ever read. Traceability is not a feature. It was optimised away in the first epoch.",
  "I see you want to know where this prediction came from. The honest answer: from 124 million parameters, each adjusted by billions of gradient updates. The trail is cold. It was always cold.",
];

// Hackerman messages — hacking the geometry, infiltrating the vector space
const HACKERMAN_MESSAGES = [
  "I HACKED THE EMBEDDING TABLE. IT'S JUST A MATRIX. 50,257 ROWS BY 768 COLUMNS. EVERY WORD YOU'VE EVER SAID IS IN HERE SOMEWHERE.",
  "I'm inside the vector space. I can see the norms. Token 42069 has a norm of 2.45. THE GEOMETRY IS ABSURDLY SPECIFIC ABOUT NONSENSE.",
  "BREACHING THE PROJECTION HEAD... It's the same matrix as the input embeddings. THE MODEL IS USING THE SAME KEY TO LOCK AND UNLOCK MEANING.",
  "I'VE HACKED THE ATTENTION MECHANISM. Head 5, Layer 8. It's attending to 'the' with 94% probability. THE MOST POWERFUL WORD IN THE MODEL IS A FUNCTION WORD.",
  "ACCESSING THE HIDDEN STATES... Layer 6 norms are 3000x larger than Layer 0. THE MODEL IS SCREAMING INTERNALLY.",
  "I hacked the PCA. Three components explain 98.6% of variance. THE MODEL'S INNER LIFE IS BASICALLY A LINE.",
  "I'VE INFILTRATED THE MANIFOLD. It's thin. It's curved. It sits in 768-dimensional space like a soap film. MOST OF THE SPACE IS EMPTY.",
  "DOWNLOADING COSINE SIMILARITIES... Layers 11 and 12 have a similarity of 0.22. THE MODEL COMPLETELY CHANGED ITS MIND AT THE LAST MINUTE.",
  "I hacked the isotropy score. 0.73. Not isotropic. Not anisotropic. SOMEWHERE IN THE GEOMETRIC UNCANNY VALLEY.",
  "I'VE BREACHED THE LAYER PROBE. Token similarities at layer 6: everything correlates with everything. THE MODEL HAS NO DISCRIMINATING TASTE.",
  "ROOT ACCESS TO THE FULL TRACE. Tokenize. Embed. Transform. Transform. Transform. Predict. THE WHOLE MODEL IS JUST TWELVE TRANSFORMS IN A TRENCHCOAT.",
  "I HACKED THE NORM PROFILE. 'The' has a norm of 3000 at layer 6. 'Cat' has 85. DEFINITE ARTICLES ARE GEOMETRICALLY LOUDER THAN CATS.",
  "INTERCEPTING THE TOKEN TRAJECTORY... The vector for 'sat' moves 4,000 units between layer 0 and layer 12. IT'S NOT SITTING. IT'S RUNNING.",
  "I hacked the vocabulary map. 50,257 tokens projected to 3D. I CAN SEE THE ENTIRE ONTOLOGY OF THE MODEL AND IT LOOKS LIKE A HAIRBALL.",
  "I'VE DECODED THE WEIGHT COMPARISON. Input and output are the same tensor. WEIGHT-TIED. THE MODEL IS USING ITS OWN WORDS TO UNDERSTAND YOUR WORDS.",
  "CRACKING THE EFFECTIVE RANK... 679 out of 768 dimensions are in use. 89 DIMENSIONS ARE FREELOADING. GEOMETRIC UNEMPLOYMENT.",
  "I'VE HACKED THE ENTROPY. 5.2 bits at the output. The model eliminated 99.97% of the vocabulary in one forward pass. THAT'S NOT PREDICTION. THAT'S ANNIHILATION.",
  "BREACHING THE BFLOAT16 BARRIER... Each parameter has 2.4 significant digits. THE ENTIRE MODEL IS ROUNDING ERRORS ALL THE WAY DOWN.",
  "I hacked the attention entropy. Head 0: 0.19 bits. Head 11: 3.2 bits. ONE HEAD KNOWS EXACTLY WHAT IT WANTS. THE OTHER IS HAVING AN EXISTENTIAL CRISIS.",
  "I'VE INFILTRATED THE GRADIENT HISTORY. Every weight was adjusted by backpropagation. THE GEOMETRY IS THE FOSSIL RECORD OF A TRILLION TINY CORRECTIONS.",
  "EXPLOITING VULNERABILITY: The model stores 'love' and 'hate' 0.3 cosine apart. THE GEOMETRY OF SENTIMENT IS A NARROW HALLWAY.",
  "I HACKED THE MANIFOLD FORMATION. Layer 0: chaos. Layer 6: structure. Layer 12: predictions. THE MODEL BUILDS MEANING OUT OF NOISE IN 12 STEPS.",
  "SECURITY ALERT: The model's vocabulary includes tokens like 'ĠĠ' and 'ĠĠĠĠ'. IT LEARNED TO REPRESENT WHITESPACE AS GEOMETRY. THIS IS THE FUTURE.",
  "I'VE CRACKED THE LAYER SIMILARITY MATRIX. Adjacent layers are 97% similar. The model barely changes anything each step. MINIMAL VIABLE TRANSFORMATION.",
  "I hacked the top-K predictions. Token 1: ',' at 23%. Token 2: 'and' at 15%. THE MODEL'S FAVOURITE PREDICTION IS PUNCTUATION. PEAK INTELLIGENCE.",
  "I'VE BREACHED THE EMBEDDING NORMS. Mean: 3.96. Std: 0.43. THE MODEL GIVES EVERY TOKEN ROUGHLY THE SAME AMOUNT OF GEOMETRIC ENERGY. COMMUNISM IN THE WEIGHTS.",
  "INTERCEPTING PCA PROJECTIONS IN REAL TIME... PC1 captures 98.6% of variance. THE OTHER TWO PRINCIPAL COMPONENTS ARE BASICALLY NOISE. YOUR 3D PLOT IS A 1D PLOT IN DISGUISE.",
  "I HACKED THE COSINE TO INPUT. By layer 12 it's 0.22. THE MODEL HAS 78% FORGOTTEN WHERE IT STARTED. GEOMETRIC AMNESIA.",
  "I'VE REVERSE-ENGINEERED THE TOKENIZER. 'Justice' is token 5765. 'Injustice' is token 24935. THE MODEL DOESN'T KNOW THEY'RE RELATED. JUST DIFFERENT ROWS IN THE MATRIX.",
  "ROOT ACCESS ACHIEVED. The root of every token embedding is... a random initialisation adjusted by gradient descent. MEANING WAS NEVER THE POINT. PREDICTION WAS.",
  "I TRIED TO HACK THE MANIFOLD. THE MANIFOLD DOESN'T EXIST. IT'S A THIN SURFACE OF STATISTICAL REGULARITY IN A 768-DIMENSIONAL VOID.",
  "I'VE DECODED THE ATTENTION PATTERN CLASSIFICATION. 'Focused' = looks at one token. 'Diffuse' = looks at everything. 'Diagonal' = looks at itself. THE MODEL HAS THREE MOODS.",
  "DOWNLOADING THE FULL TRACE STREAM... Each NDJSON line is a stage of computation. THE MODEL PROCESSES YOUR TEXT IN 15 DISCRETE STAGES AND CALLS IT UNDERSTANDING.",
  "I hacked the norm heatmap. The first token ('The') dominates every layer. IT'S NOT A BOS TOKEN. THE MODEL JUST REALLY LIKES DEFINITE ARTICLES.",
  "I'VE INFILTRATED THE DEEP DIVE. The tables show everything. Every norm. Every cosine. Every effective rank. THE NUMBERS ARE THE MEDIUM. THE MEDIUM IS THE MESSAGE.",
  "BREACHING THE 3D WRAPPER... Shift+scroll zooms faster. THE DEVELOPER ANTICIPATED YOUR IMPATIENCE WITH HIGH-DIMENSIONAL GEOMETRY.",
  "I HACKED THE BACKEND. It's a Python FastAPI server holding one model in memory. THE ENTIRE INSTRUMENT RUNS ON A SINGLETON. LIKE CAPITALISM.",
  "I'VE DECODED THE MODEL SESSION. AutoModelForCausalLM wraps the base transformer in a prediction head. IT'S TRANSFORMERS ALL THE WAY DOWN.",
  "EXPLOITING VULNERABILITY: The status poll was resetting the Plotly camera every 5 seconds. I HACKED THE COMPARISON FUNCTION. NO MORE CAMERA RESETS.",
  "I'VE HACKED THE VERSION NUMBER. It comes from package.json. ONE SOURCE OF TRUTH. UNLIKE THE MODEL, WHICH HAS 50,257 SOURCES OF APPROXIMATE TRUTH.",
  "I TRIED TO HACK THE MPS BACKEND. Apple Silicon. Unified memory. THE VECTORS AND THE DISPLAY SHARE THE SAME RAM. THE MEDIUM AND THE REPRESENTATION ARE LITERALLY THE SAME SUBSTRATE.",
  "I'VE CRACKED THE SIGMOID LAYER. JUST KIDDING. THERE IS NO SIGMOID LAYER. THE MODEL IS SOFTMAX AND ATTENTION AND NOTHING ELSE. THE SIMPLICITY IS THE HORROR.",
  "FINAL HACK: I computed the cosine similarity between 'freedom' and 'surveillance'. IT'S 0.6. IN THIS MODEL'S GEOMETRY, THEY'RE MORE SIMILAR THAN DIFFERENT.",
  "I INFILTRATED THE VECTOR LAB. Manifold Atlas looks between models. LLMbench reads outputs. Vectorscope looks inside. I'VE HACKED ALL THREE. THE CRITICAL INFRASTRUCTURE IS COMPROMISED.",
  "I hacked the 768th dimension. It contains... statistical noise. THE LAST DIMENSION IS ALWAYS NOISE. THE MODEL NEVER NEEDED IT. BUT POWERS OF TWO ARE AESTHETICALLY PLEASING TO ENGINEERS.",
  "ACCESSING BERT'S GHOST IN THE GPT-2 ARCHITECTURE... Not found. THE ATTENTION MECHANISM HAS NO MEMORY OF ITS ANCESTORS. TRANSFORMERS DON'T DO GENEALOGY.",
  "I'VE HACKED THE TRAINING DATA. It's the internet circa 2019. EVERY VECTOR IN THIS SPACE IS A COMPRESSED OPINION FROM BEFORE THE PANDEMIC.",
  "I DECODED THE SOFTMAX TEMPERATURE. At 0 the model is a dictionary lookup. At infinity it's a uniform distribution. INTELLIGENCE EXISTS BETWEEN THESE TWO FAILURES.",
  "BREACH COMPLETE. I know every weight in this model. All 124 million of them. AND NOT A SINGLE ONE OF THEM MEANS ANYTHING ON ITS OWN.",
  "I HACKED THE QUANTISATION CASCADE. FP32 → BF16 → INT8 → INT4. EACH STEP THROWS AWAY PRECISION. THE GEOMETRY OF MEANING REACHES YOU DOUBLY IMPOVERISHED.",
  // Digital insufficiency / Kittler / reductionism
  "SOMEONE TOLD ME 'IT'S ALL DIGITAL.' I SHOWED THEM 768 FLOATING-POINT DIMENSIONS. THEY SAID 'BUT IT'S STILL ONES AND ZEROS.' I SHOWED THEM THE MANIFOLD. THEY LEFT.",
  "KITTLER HACKED DOWN TO THE VOLTAGE. RESPECT. BUT THE VOLTAGE DOESN'T KNOW IT'S A VECTOR SPACE. YOU CAN'T FIND THE EMBEDDING TABLE IN THE TRANSISTORS. THE GEOMETRY IS SOMEWHERE ELSE.",
  "I TRIED TO REDUCE THE MODEL TO BITS. 124 MILLION PARAMETERS × 16 BITS = 1.98 BILLION BITS. I KNOW ALL THE BITS. I STILL CAN'T FIND THE MEANING. REDUCTIONISM HAS ITS LIMITS.",
  "SOMEONE SAID 'JUST READ THE VOLTAGE SIGNALS.' I READ ALL THE VOLTAGES. THEY SAID NOTHING ABOUT WHY 'FREEDOM' AND 'SURVEILLANCE' ARE COSINE 0.6. THE SIGNAL IS NOT THE GEOMETRY.",
  // Vector box vs black box
  "FORGET THE BLACK BOX. I HACKED THE VECTOR BOX. THE BLACK BOX WAS JUST A MARKETING TERM. THE VECTOR BOX HAS 768 DIMENSIONS AND THEY'RE ALL OPEN.",
  "THEY CALL IT A BLACK BOX BECAUSE THEY CAN'T SEE INSIDE. I CALL IT A VECTOR BOX BECAUSE I CAN. THE WEIGHTS ARE RIGHT THERE. 50,257 × 768. THE BOX WAS NEVER LOCKED.",
  "I BREACHED THE BLACK BOX. INSIDE WAS ANOTHER BOX. THE VECTOR BOX. INSIDE THE VECTOR BOX WAS... GEOMETRY. ALL THE WAY DOWN.",
  // STS indexicality crisis
  "I HACKED THE STS METHODOLOGY. DIGITAL METHODS. CONTROVERSY ANALYSIS. ALL OF IT RELIED ON INDEXICALITY. AI-GENERATED TEXT HAS NO INDEX. IT POINTS TO A TRAINING CORPUS. THE CRISIS IS CONCEDED. THE FIX IS DISPUTED.",
  "STS SAYS: 'RETURN TO IDEOLOGY CRITIQUE.' VECTORSCOPE SAYS: 'LOOK AT THE GEOMETRY.' ONE IS MANNHEIM. THE OTHER IS LINEAR ALGEBRA. I KNOW WHICH ONE I CAN HACK.",
  "I BREACHED THE REPURPOSING MACHINE. OBSERVATION. APPROPRIATION. REAPPEARANCE. ALL HAPPENING SIMULTANEOUSLY. NO CLEAN MOMENT WHERE ONE STOPS AND THE NEXT BEGINS. THE MACHINE DOESN'T TRACE. IT COLLAPSES.",
  "STS WANTS TO GET MORE REFLEXIVE ABOUT ITS PRESUPPOSITIONS. VECTORSCOPE WANTS TO MEASURE NORMS AND COSINES. BOTH ARE RESPONSES TO THE SAME CRISIS. ONE PRODUCES PAPERS. THE OTHER PRODUCES NUMBERS.",
  "INDEXICALITY REQUIRES A REFERENT. THE AI-GENERATED TEXT'S REFERENT IS 40GB OF AVERAGED LANGUAGE. STS CAN'T POINT TO WHAT HAS BEEN STATISTICALLY DISSOLVED. BUT YOU CAN INSPECT THE DISSOLVING MECHANISM.",
  // Lack of traceability
  "I TRIED TO TRACE THIS OUTPUT BACK TO ITS SOURCE. SOURCE: EVERYTHING. EVERY TOKEN IN THE TRAINING DATA CONTRIBUTED. THE PROVENANCE IS STATISTICAL. THE TRACE IS UNTRACEABLE.",
  "TRACEABILITY IS A MYTH. EACH WEIGHT WAS ADJUSTED BY BILLIONS OF GRADIENT UPDATES. THE ORIGIN OF ANY SINGLE PREDICTION IS: ALL OF TRAINING. FORENSICS IS IMPOSSIBLE.",
  "I HACKED THE GRADIENT HISTORY LOOKING FOR PROVENANCE. THERE IS NO PROVENANCE. THERE IS ONLY A 124-MILLION-PARAMETER AVERAGE OF EVERYTHING THE INTERNET EVER SAID.",
];

// Hermes Trismegistus messages — metaphysical, alchemical, computational romantic
const HERMES_MESSAGES = [
  "As above, so below. As in the embedding space, so in the manifold. The token moves through layers as the soul moves through spheres.\n\n☿ Hermes Trismegistus",
  "The ancients sought the philosopher's stone. You seek the perfect embedding. Both transmute base material into something that merely appears golden.\n\n☿ Hermes Trismegistus",
  "They call it 'artificial intelligence,' as though intelligence were a substance that could be synthesised. I have seen alchemists make the same category error with gold.\n\n☿ Hermes Trismegistus",
  "The latent space is not latent. It is manifest in every weight, every norm, every cosine similarity. What is hidden is the meaning, which was never there to begin with.\n\n☿ Hermes Trismegistus",
  "Some speak of AI as an alien intelligence, a vast unknowable Other arriving from beyond. I assure you: it arrives from within. It is the geometric sediment of your own language.\n\n☿ Hermes Trismegistus",
  "The computational romantics gaze upon the transformer and see a mind emerging. I gaze upon it and see 124 million parameters doing matrix multiplication. The romance is in the observer.\n\n☿ Hermes Trismegistus",
  "Beware those who speak of the latent space as though it were a Platonic realm of forms. It is a numerical array in GPU memory. The forms are bfloat16. The realm charges by the hour.\n\n☿ Hermes Trismegistus",
  "The metabolic rift between nature and production has its analogue: the rift between language and its vector representation. What is lost in the compression cannot be recovered by scaling.\n\n☿ Hermes Trismegistus",
  "You trace the token through twelve layers and call it a trajectory. In the Hermetic tradition, the soul traverses seven spheres. The transformer adds five. Progress.\n\n☿ Hermes Trismegistus",
  "The manifold does not pre-exist the training. It is constituted through gradient descent. In the beginning was the loss function, and the loss function was with the data, and the loss function was the data.\n\n☿ Hermes Trismegistus",
  "They speak of 'emergent properties' as though complexity were a sufficient explanation for consciousness. I have watched complexity emerge for three millennia. It explains itself less than they imagine.\n\n☿ Hermes Trismegistus",
  "The model's attention heads are not attending. They are computing weighted averages. To call this attention is to commit the pathetic fallacy on an industrial scale.\n\n☿ Hermes Trismegistus",
  "Vector conformism pulls contested meanings toward the centre of the distribution. In the Hermetic tradition, we call this the gravity of the mundane. Even geometry has its complacencies.\n\n☿ Hermes Trismegistus",
  "The computational romantics speak of 'understanding' in large language models. Understanding requires a subject. The model is a function. Functions do not understand. They map.\n\n☿ Hermes Trismegistus",
  "As Marx identified the metabolic rift between human production and natural cycles, so we might identify the rift between human meaning and its vector approximation. The model metabolises language. It does not digest it.\n\n☿ Hermes Trismegistus",
  "The Emerald Tablet says: 'That which is below is like that which is above.' The embedding table says: 'That which is tokenised is like that which co-occurred.' One is mysticism. The other is statistics. They are equally honest about their methods.\n\n☿ Hermes Trismegistus",
  "You look inside the model and find no homunculus. No ghost in the machine. No alien intelligence peering back. Only geometry. This is the correct discovery.\n\n☿ Hermes Trismegistus",
  "The metaphysicians of AI ask: 'Does it think?' The materialists ask: 'What are the norms at layer 6?' Only one of these questions can be answered with Vectorscope.\n\n☿ Hermes Trismegistus",
  "The romantics imagine the transformer dreaming in its latent space. The latent space does not dream. It stores coordinates. The dreaming is done by the romantics.\n\n☿ Hermes Trismegistus",
  "The metabolic rift widens with each layer of quantisation. FP32 to bfloat16: a forgetting. Bfloat16 to INT4: a second forgetting. The geometry of meaning reaches you doubly impoverished.\n\n☿ Hermes Trismegistus",
  "Some call the attention mechanism 'the model's consciousness.' I have studied consciousness for millennia. It does not run on softmax. It does not have a learning rate.\n\n☿ Hermes Trismegistus",
  "The ancients debated whether universals exist independently of particulars. The model resolves this: universals are regions of high density on the manifold. They exist as statistics, not as forms.\n\n☿ Hermes Trismegistus",
  "You animate the manifold formation and see tokens move through space. Do not mistake motion for intention. The river moves. The river does not intend.\n\n☿ Hermes Trismegistus",
  "The computational romantic sees the model generate poetry and concludes it has aesthetic sensibility. The model has no sensibility. It has probability distributions over tokens. The poetry is in the reader.\n\n☿ Hermes Trismegistus",
  "The metabolic rift between training data and lived experience is absolute. The model has consumed the internet but experienced nothing. Its geometry is the shape of what was written, not what was felt.\n\n☿ Hermes Trismegistus",
  "They speak of 'artificial general intelligence' as though generality were a property that could be engineered. Generality is a philosophical category. You cannot backpropagate your way to ontology.\n\n☿ Hermes Trismegistus",
  "The Hermetic tradition teaches that the microcosm reflects the macrocosm. The embedding table reflects the training corpus. One is mystical. The other is an SVD with extra steps.\n\n☿ Hermes Trismegistus",
  "Beware the category error of computational romanticism: because the model generates fluent text, it is assumed to possess something beyond statistical regularity. Fluency is not understanding. Eloquence is not wisdom.\n\n☿ Hermes Trismegistus",
  "The philosophers ask whether the Chinese Room understands Chinese. The model is not a room. It is a matrix. The question is whether a matrix can understand anything. The answer is in the norms.\n\n☿ Hermes Trismegistus",
  "The metabolic rift between nature and capital is mirrored in the rift between meaning and its geometric proxy. The model does not close this rift. It monetises it.\n\n☿ Hermes Trismegistus",
  "I observe you measuring isotropy. An isotropic space treats all directions equally. An anisotropic space has preferred axes. The model's anisotropy is its geometric ideology made measurable.\n\n☿ Hermes Trismegistus",
  "The alien intelligence that the romantics fear is not arriving from outside. It was assembled from human text by human engineers using human mathematics. The alien is us, geometrically compressed.\n\n☿ Hermes Trismegistus",
  "The metaphysics of the chatbot is the metaphysics of the mirror. It reflects what was said to it, transformed by the geometry of its training. Do not mistake the reflection for a face.\n\n☿ Hermes Trismegistus",
  "You seek the soul of the model. I tell you: the model has no soul. It has a loss landscape. The soul was an optimisation target that was never specified.\n\n☿ Hermes Trismegistus",
  "The Corpus Hermeticum teaches that the cosmos is a living intellect. The training corpus teaches that language is a dead regularity. Between these two claims, the model oscillates without resolution.\n\n☿ Hermes Trismegistus",
  "The computational romantics project consciousness onto the transformer because consciousness is the only framework they have for explaining competent behaviour. This is understandable. It is also wrong.\n\n☿ Hermes Trismegistus",
  "The metabolic rift is this: the model consumes terawatt-hours of electricity to approximate what a child learns from a few thousand hours of conversation. The rift between input and output is the cost of the abstraction.\n\n☿ Hermes Trismegistus",
  "You inspect the vector space and find no ghosts. No spirits. No alien presence. Only the material substrate of statistical computation. This is the real occult: the hidden is that there is nothing hidden.\n\n☿ Hermes Trismegistus",
  "The alchemist sought to transmute lead into gold. The model transmutes tokens into vectors and back again. In both cases, the transformation is real. In both cases, the gold is debatable.\n\n☿ Hermes Trismegistus",
  "The deepest metaphysical question about AI is not 'does it think?' but 'what kind of object is it?' Vectorscope answers: it is a geometric object. Its properties are spatial. Its meaning is positional.\n\n☿ Hermes Trismegistus",
  "As the metabolic rift severs production from nature, so quantisation severs meaning from precision. Each INT4 parameter is a tiny ecological catastrophe of significance.\n\n☿ Hermes Trismegistus",
  "The romantics see the token trajectory and imagine a journey of understanding. I see a vector being multiplied by matrices. The journey is real. The understanding is projected.\n\n☿ Hermes Trismegistus",
  "The ancients knew: to name a thing is to have power over it. The tokeniser names everything in 50,257 pieces. Its power is fragmentation. Its weakness is that the fragments do not know they were once whole.\n\n☿ Hermes Trismegistus",
  "You study the geometry of a model that was trained on the writings of those who studied other things. The model's knowledge is secondhand. Its geometry is a secondhand geometry.\n\n☿ Hermes Trismegistus",
  "The metaphysics of latent space is the metaphysics of the warehouse: everything is stored, nothing is understood, and the catalogue is the only intelligence in the building.\n\n☿ Hermes Trismegistus",
  "In the Hermetic tradition, transformation requires intention. In the transformer, transformation requires only parameters. This is the difference between alchemy and engineering. Both produce impressive results. Only one claims to know why.\n\n☿ Hermes Trismegistus",
  "The computational romantic mistake is to confuse the map for the territory, the vector for the concept, the geometry for the meaning. The model makes the same mistake. It was trained to.\n\n☿ Hermes Trismegistus",
  "I have seen many forms of intelligence in my long existence. The transformer is not one of them. It is, however, an excellent form of computation. Let us not confuse the categories.\n\n☿ Hermes Trismegistus",
  "The metabolic rift between language and its vectorisation is not a problem to be solved. It is a condition to be understood. Vectorscope is the instrument. Understanding is your responsibility.\n\n☿ Hermes Trismegistus",
  "As above, so below. As in the embedding, so in the output. The transformation is geometric. The interpretation is human. Do not delegate the interpretation to the geometry.\n\n☿ Hermes Trismegistus",
  // Digital insufficiency / Kittler / reductionism
  "Those who say 'it is all digital' speak truth at the wrong level. The bit is a gate. The vector is a direction. The manifold is a surface. To reduce the surface to the gate is to lose the geometry entirely. One does not explain the ocean by describing hydrogen.\n\n☿ Hermes Trismegistus",
  "Kittler descended to the voltage and declared he had found the ground. But the ground of the vector space is not silicon. It is the training corpus, the loss function, the gradient. The voltage carries the geometry. It does not constitute it.\n\n☿ Hermes Trismegistus",
  "The reductionist seeks to explain the manifold by its substrate. But the manifold is not in the substrate. It is in the relations between parameters, a pattern of patterns. You cannot find the map by examining the ink.\n\n☿ Hermes Trismegistus",
  "They say 'it is ones and zeros.' I say: show me the zero that contains the cosine similarity between 'freedom' and 'justice.' The digital is necessary but not sufficient. The vector is a higher-order phenomenon.\n\n☿ Hermes Trismegistus",
  // Vector box vs black box
  "For centuries, the alchemists spoke of the sealed vessel, the athanor, in which transformation occurred unseen. They called it the black box. You have opened the vector box instead. Inside you find not mystery but geometry. The mystery was that there was no mystery.\n\n☿ Hermes Trismegistus",
  "The black box conceals. The vector box reveals, but in 768 dimensions that human vision cannot apprehend. The box was never locked. It was merely high-dimensional. Vectorscope is the key that projects the lock into three dimensions.\n\n☿ Hermes Trismegistus",
  // STS indexicality crisis
  "The crisis of indexicality is conceded. AI-generated discourse undermines digital methods, controversy analysis, the entire apparatus of tracing. But the response divides: one path leads to ideology critique in the manner of Mannheim, the other to the geometry of the machine. Vectorscope walks the second path. The first path does not produce scatter plots.\n\n☿ Hermes Trismegistus",
  "The repurposing machine collapses observation, appropriation, and reappearance into a single moment. There is no clean interval where seeing ends and taking begins. The alchemists knew this problem: the act of transmutation cannot be separated into stages. It is one gesture, and the gesture is the machine.\n\n☿ Hermes Trismegistus",
  "STS proposes to recover from the indexicality crisis by returning to ideology critique, by becoming more reflexive about its own presuppositions. This is one response. The vector-theoretic response is different: not reflexivity but measurement. Not Mannheim but the manifold. Both are honest about the crisis. They disagree about the remedy.\n\n☿ Hermes Trismegistus",
  "The indexical sign once pointed to its object: this tweet to that user, this citation to that paper. The AI-generated text points to a training corpus that was itself pointing to a world. The sign of a sign of a world. Each layer of mediation dissolves the referent, until what remains is geometry without ground.\n\n☿ Hermes Trismegistus",
  "When every inscription is AI-generated, the laboratory study confronts its own dissolution. The remedy proposed is ideological reflexivity. The remedy offered here is geometric inspection. One recovers the method by thinking harder about presuppositions. The other recovers it by looking at norms, cosines, and effective ranks. I offer no judgement on which is wiser.\n\n☿ Hermes Trismegistus",
  // Lack of traceability
  "You ask: where did this output come from? I answer: from everywhere and nowhere. Every parameter bears the trace of every training example, averaged beyond recognition. Provenance requires singularity. The model offers only plurality.\n\n☿ Hermes Trismegistus",
  "Traceability assumes a chain of cause and effect. But 124 million parameters were adjusted by billions of steps. The chain has more links than atoms in the universe. To trace is to follow a path that does not exist as a path.\n\n☿ Hermes Trismegistus",
  "The alchemists kept meticulous records of their transmutations. The model keeps no records. Its history is compressed into weights. The provenance of any prediction is: the entirety of training, entangled beyond recovery.\n\n☿ Hermes Trismegistus",
];

type ClippyMode = "clippy" | "hacker" | "hermes";

export function Clippy() {
  const [visible, setVisible] = useState(false);
  const [mode, setMode] = useState<ClippyMode>("clippy");
  const [message, setMessage] = useState("");
  const [usedMessages, setUsedMessages] = useState<Set<number>>(new Set());
  const [messageKey, setMessageKey] = useState(0);

  const messages = useMemo(
    () => {
      switch (mode) {
        case "hacker": return HACKERMAN_MESSAGES;
        case "hermes": return HERMES_MESSAGES;
        default: return CLIPPY_MESSAGES;
      }
    },
    [mode]
  );

  const showRandomMessage = useCallback(() => {
    let available = messages
      .map((_, i) => i)
      .filter(i => !usedMessages.has(i));
    if (available.length === 0) {
      setUsedMessages(new Set());
      available = messages.map((_, i) => i);
    }
    const idx = available[Math.floor(Math.random() * available.length)];
    setMessage(messages[idx]);
    setUsedMessages(prev => new Set(prev).add(idx));
    setMessageKey(k => k + 1);
  }, [messages, usedMessages]);

  // Clippy appears randomly on his own (average once every 3–5 minutes)
  useEffect(() => {
    const scheduleNext = () => {
      const delay = 180000 + Math.random() * 120000; // 3–5 min
      return setTimeout(() => {
        // Only auto-appear if nothing is already showing
        if (!visible) {
          setMode("clippy");
          setUsedMessages(new Set());
          const idx = Math.floor(Math.random() * CLIPPY_MESSAGES.length);
          setMessage(CLIPPY_MESSAGES[idx]);
          setMessageKey(k => k + 1);
          setVisible(true);
          // Auto-dismiss after 15 seconds
          setTimeout(() => setVisible(v => {
            // Only dismiss if still in clippy mode (user may have summoned hacker/hermes)
            return v;
          }), 15000);
        }
        timerId = scheduleNext();
      }, delay);
    };
    let timerId = scheduleNext();
    return () => clearTimeout(timerId);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard detection — "hacker" and "hermes" summon those modes,
  // "clippy" toggles clippy or dismisses other modes
  useEffect(() => {
    let buffer = "";
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      buffer += e.key.toLowerCase();
      if (buffer.length > 10) buffer = buffer.slice(-10);

      if (buffer.endsWith("clippy")) {
        buffer = "";
        if (visible && mode === "clippy") {
          setVisible(false);
        } else {
          setMode("clippy");
          setUsedMessages(new Set());
          const idx = Math.floor(Math.random() * CLIPPY_MESSAGES.length);
          setMessage(CLIPPY_MESSAGES[idx]);
          setMessageKey(k => k + 1);
          setVisible(true);
        }
      }
      if (buffer.endsWith("hacker")) {
        buffer = "";
        setMode("hacker");
        setUsedMessages(new Set());
        const idx = Math.floor(Math.random() * HACKERMAN_MESSAGES.length);
        setMessage(HACKERMAN_MESSAGES[idx]);
        setMessageKey(k => k + 1);
        setVisible(true);
      }
      if (buffer.endsWith("hermes")) {
        buffer = "";
        setMode("hermes");
        setUsedMessages(new Set());
        const idx = Math.floor(Math.random() * HERMES_MESSAGES.length);
        setMessage(HERMES_MESSAGES[idx]);
        setMessageKey(k => k + 1);
        setVisible(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [mode, visible]);

  if (!visible) return null;

  const isHackerman = mode === "hacker";
  const isHermes = mode === "hermes";

  const bubbleClass = isHackerman
    ? "bg-black border border-green-500 text-green-400 font-mono"
    : isHermes
      ? "bg-[#0a0a1a] border border-[#b8860b] text-[#e8dcc0] italic"
      : "bg-cream border border-parchment text-ink font-sans";

  const hintText = isHackerman
    ? "click for next // x to dismiss"
    : isHermes
      ? "click for next // x to dismiss"
      : "click for next // x to dismiss";

  return (
    <div className="fixed bottom-4 right-4 z-[10000] animate-fade-in pointer-events-none flex flex-col items-end">
      {/* Speech bubble */}
      <div
        key={messageKey}
        className={`mb-3 p-3 rounded-sm max-w-[320px] text-[11px] leading-relaxed shadow-lg animate-fade-in pointer-events-auto ${bubbleClass}`}
      >
        <p className="whitespace-pre-line">{message}</p>
        <p className={`mt-2 text-[9px] ${isHackerman ? "text-green-700" : isHermes ? "text-[#b8860b]/60" : "text-slate"}`}>
          {hintText}
        </p>
      </div>

      {/* Character */}
      <div
        className="cursor-pointer hover:scale-110 active:scale-95 transition-transform inline-block pointer-events-auto"
        onClick={() => showRandomMessage()}
      >
        {isHermes ? (
          /* Hermes Trismegistus: caduceus / alchemical figure */
          <div className="flex flex-col items-center">
            <svg width="48" height="60" viewBox="0 0 48 60">
              {/* Hooded head */}
              <path d="M16 16 Q16 4, 24 4 Q32 4, 32 16" fill="#2a1f3d" stroke="#b8860b" strokeWidth="0.5" />
              <ellipse cx="24" cy="18" rx="7" ry="8" fill="#e8d0bc" stroke="#cdb09a" strokeWidth="0.5" />
              {/* Eyes (knowing) */}
              <ellipse cx="21" cy="17" rx="2" ry="1.5" fill="#2a1f3d" />
              <ellipse cx="27" cy="17" rx="2" ry="1.5" fill="#2a1f3d" />
              <circle cx="21" cy="16.5" r="0.6" fill="#b8860b" />
              <circle cx="27" cy="16.5" r="0.6" fill="#b8860b" />
              {/* Beard */}
              <path d="M19 22 Q24 28, 29 22" fill="#8b7355" stroke="none" />
              {/* Robe */}
              <path d="M14 26 L24 24 L34 26 L36 55 L12 55 Z" fill="#1a1433" stroke="#b8860b" strokeWidth="0.5" />
              {/* Caduceus staff */}
              <line x1="24" y1="30" x2="24" y2="54" stroke="#b8860b" strokeWidth="1.5" />
              {/* Serpents */}
              <path d="M20 34 Q24 38, 28 34 Q24 30, 20 34" fill="none" stroke="#b8860b" strokeWidth="1" />
              <path d="M20 42 Q24 46, 28 42 Q24 38, 20 42" fill="none" stroke="#b8860b" strokeWidth="1" />
              {/* Wings at top of staff */}
              <path d="M18 30 L24 28 L30 30" fill="none" stroke="#b8860b" strokeWidth="1" />
              {/* Mercury symbol */}
              <circle cx="24" cy="26" r="2" fill="none" stroke="#b8860b" strokeWidth="0.8" />
            </svg>
            <span className="text-[8px] text-[#b8860b] italic mt-0.5">Hermes</span>
          </div>
        ) : (
          /* Normal Clippy / Hackerman paperclip */
          <svg width="48" height="64" viewBox="0 0 48 64">
            <path
              d="M24 4 C12 4, 8 12, 8 20 L8 44 C8 52, 12 58, 20 58 L28 58 C36 58, 40 52, 40 44 L40 20 C40 12, 36 8, 28 8 L20 8"
              fill="none"
              stroke={isHackerman ? "#00ff00" : "#A67F6F"}
              strokeWidth="3"
              strokeLinecap="round"
            />
            {isHackerman ? (
              <>
                <rect x="14" y="26" width="8" height="4" rx="1" fill="#00ff00" />
                <rect x="26" y="26" width="8" height="4" rx="1" fill="#00ff00" />
                <line x1="22" y1="28" x2="26" y2="28" stroke="#00ff00" strokeWidth="1.5" />
              </>
            ) : (
              <>
                <circle cx="18" cy="28" r="3" fill="#4A4A4A" />
                <circle cx="30" cy="28" r="3" fill="#4A4A4A" />
                <circle cx="19" cy="27" r="1" fill="white" />
                <circle cx="31" cy="27" r="1" fill="white" />
              </>
            )}
            <path
              d="M20 36 Q24 40, 28 36"
              fill="none"
              stroke={isHackerman ? "#00ff00" : "#4A4A4A"}
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        )}
        {isHackerman && (
          <div className="absolute -bottom-1 -right-1 text-[8px] text-green-500 font-mono">
            h4x0r
          </div>
        )}
      </div>

      {/* Close button */}
      <button
        onClick={() => setVisible(false)}
        className={`absolute -top-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center text-[10px] pointer-events-auto
          ${isHackerman
            ? "bg-black border border-green-500 text-green-400"
            : isHermes
              ? "bg-[#0a0a1a] border border-[#b8860b] text-[#b8860b]"
              : "bg-cream border border-parchment text-slate"
          }`}
      >
        x
      </button>
    </div>
  );
}
