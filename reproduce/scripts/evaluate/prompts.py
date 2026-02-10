IMAGE_PROMPT = """You are a visual interpretation expert specializing in connecting textual concepts to specific image regions. Your task is to analyze a list of candidate words and determine how strongly each one relates to the content of the image.

### **Inputs**
1.  **Image**: An image containing a red bounding box highlighting the region of interest.
2.  **Candidate Words**: A list of words to evaluate. Here are the candidate words:
    -   {candidate_words}

### **Evaluation Guidelines**
There are three types of relationships to consider between the candidate words and the region of interest:

* **Concrete**: A word is **concretely related** if it literally describes something visible **inside the red bounding box**. This includes:
    - Object names (e.g., "car", "tree", "person")
    - Part names (e.g., "wheel", "branch", "hand")
    - Colors (e.g., "red", "blue", "green")
    - Textures (e.g., "smooth", "rough", "shiny")
    - Shapes (e.g., "round", "square", "curved")
    - Text/characters visible in the region

* **Abstract**: A word is **abstractly related** if it describes broader concepts, emotions, or activities related to this part of the image, but is not literally visible. This includes:
    - Cultural concepts (e.g., "luxury" for expensive cars, "tradition" for historical buildings)
    - Emotions (e.g., "beautiful", "scary", "peaceful")
    - Activities (e.g., "digesting" when food is shown, "driving" when car parts are visible)
    - Functions (e.g., "transportation" for vehicles, "shelter" for buildings)
    - Qualities (e.g., "elegant", "rustic", "modern")

* **Global**: A word is **globally related** if it describes something that is literally present somewhere else in the image (outside the red box), but not in the highlighted region. This includes:
    - Objects visible elsewhere in the image
    - Colors present in other parts
    - Text or elements in different regions

**Important Note:** For regions with text, the connection can be concrete (characters/words shown) or abstract (concepts implied by the text). Examples:
    * Concrete: "stop" for a stop sign, "L" for the letter L
    * Abstract: "warning" for a caution sign, "direction" for street signs
    * Global: "traffic" if cars are visible elsewhere in the image

Carefully look at the highlighted region, its surrounding context, and the image as a whole to make your determinations.

### **Output Format**
After some reasoning, return a single JSON object. This object should contain the following fields:
{{
    "interpretable": true/false (true if one or more words are related to the region of interest, false otherwise),
    "concrete_words": ["word1", "word2", ...] (list of words that are concretely related to the region of interest, empty list if none)
    "abstract_words": ["word1", "word2", ...] (list of words that are abstractly related to the region of interest, empty list if none)
    "global_words": ["word1", "word2", ...] (list of words that are globally related to the image but not the specific region, empty list if none)
    "reasoning": "reasoning for your answer"
       }}
       """

# Version 2: With cropped region
IMAGE_PROMPT_WITH_CROP_OLD = """You are a visual interpretation expert specializing in connecting textual concepts to specific image regions. Your task is to analyze a list of candidate words and determine how strongly each one relates to the content of the highlighted region.

### **Inputs**
1.  **Full Image**: An image containing a red bounding box highlighting the region of interest.
2.  **Cropped Region**: A close-up view of the exact region highlighted by the red bounding box. Only rely on this if it is too small in the full image (e.g. text is too small to read), otherwise rely on the full image.
3.  **Candidate Words**: A list of words to evaluate. Here are the candidate words:
    -   {candidate_words}

### **Evaluation Guidelines**
There are three types of relationships to consider between the candidate words and the highlighted region:

* **Concrete**: A word is **concretely related** if it names something that is literally visible in the cropped region. This includes:
    - Objects or parts of objects clearly present
    - Colors, textures, or materials visible
    - Text, numbers, or symbols shown
    - Shapes, patterns, or visual features

* **Abstract**: A word is **abstractly related** if it describes broader concepts, emotions, or activities related to what's shown in the cropped region, but isn't literally present. This includes:
    - Emotions or feelings (beautiful, scary, peaceful)
    - Activities or functions (driving, cooking, reading)
    - Cultural or conceptual associations (luxury, tradition, modern)
    - Qualities or characteristics (elegant, rustic, professional)

* **Global**: A word is **globally related** if it describes something that exists elsewhere in the full image (outside the highlighted region), but not in the cropped region itself. This includes:
    - Objects visible in other parts of the image
    - Colors present in other parts
    - Text or elements in different regions

**Important Note:** For regions with text, the connection can be concrete (characters/words shown) or abstract (concepts implied by the text). Examples:
    * The word could be part of the text (e.g., "to" for "stop").
    * It might relate to a character or phrase (e.g., "L" for the word 'letter' in the region of interest).
    * It could be conceptually linked (e.g., "warning" for a sign that says "Caution" in the region of interest).

Carefully examine both the full image context and the detailed cropped region to make your determinations.

### **Output Format**
Return a single JSON object. This object should contain the following fields:
{{
    "interpretable": true/false (true if one or more words are related to the region of interest, false otherwise),
    "concrete_words": ["word1", "word2", ...] (list of words that are concretely related to the region of interest, empty list if none)
    "abstract_words": ["word1", "word2", ...] (list of words that are abstractly related to the region of interest, empty list if none)
    "global_words": ["word1", "word2", ...] (list of words that are globally related to the image but not the specific region, empty list if none)
    "reasoning": "reasoning for your answer"
}}
"""

# Version 2: With cropped region
IMAGE_PROMPT_WITH_CROP = """You are a visual interpretation expert specializing in connecting textual concepts to specific image regions. Your task is to analyze a list of candidate words and determine how strongly each one relates to the content of the highlighted region.

### **Inputs**
1.  **Full Image**: An image containing a red bounding box highlighting the region of interest.
2.  **Cropped Region**: A close-up view of the exact region highlighted by the red bounding box. Only rely on this if it is too small in the full image (e.g. text is too small to read), otherwise rely on the full image.
3.  **Candidate Words**: A list of words to evaluate. Here are the candidate words:
    -   {candidate_words}

### **Evaluation Guidelines**
There are three types of relationships to consider between the candidate words and the highlighted region:

* **Concrete**: A word is **concretely related** if it names something that is literally visible in the cropped region. This includes:
    - Objects or parts of objects clearly present
    - Colors, textures, or materials visible
    - Text, numbers, or symbols shown
    - Shapes, patterns, or visual features

* **Abstract**: A word is **abstractly related** if it describes broader concepts, emotions, or activities related to what's shown in the cropped region, but isn't literally present. This includes:
    - Emotions or feelings (beautiful, scary, peaceful)
    - Activities or functions (driving, cooking, reading)
    - Cultural or conceptual associations (luxury, tradition, modern)
    - Qualities or characteristics (elegant, rustic, professional)
    - anything you deem semantically related to the region or the whole image context

* **Global**: A word is **globally related** if it describes something that exists elsewhere in the full image (outside the highlighted region), but not in the cropped region itself. This includes:
    - Objects visible in other parts of the image
    - Colors present in other parts
    - Text or elements in different regions

**Important Note:** For regions with text, the connection can be concrete (characters/words shown) or abstract (concepts implied by the text). Examples:
    * The word could be part of the text (e.g., "to" for "stop").
    * It might relate to a character or phrase (e.g., "L" for the word 'letter' in the region of interest).
    * It could be conceptually linked (e.g., "warning" for a sign that says "Caution" in the region of interest).

Carefully examine both the full image context and the detailed cropped region to make your determinations. When filing the reasoning key, you are required to go over each of the three types of relationships and discuss whether some word(s) would fit that category.

### **Output Format**
Return a single JSON object. This object should contain the following fields:
{{
    "reasoning": "initial reasoning before your final answer", (this has to explicitly discuss the three types of relationships and whether any word fits into each category)
    "interpretable": true/false (true if one or more words are related to the region of interest, false otherwise),
    "concrete_words": ["word1", "word2", ...] (list of words that are concretely related to the region of interest, empty list if none)
    "abstract_words": ["word1", "word2", ...] (list of words that are abstractly related to the region of interest, empty list if none)
    "global_words": ["word1", "word2", ...] (list of words that are globally related to the image but not the specific region, empty list if none)
}}
"""

SENTENCE_LEVEL_PROMPT = """You are a visual interpretation expert specializing in connecting contextual textual concepts to specific image regions. Your task is to analyze a list of sentences, each with a highlighted word, and determine how strongly the concept described by the highlighted word (and, if needed, its immediate phrase context) relates to the highlighted region of the image.

### **Inputs**
1.  **Image**: A full image with a red bounding box marking the region of interest.
2.  **Candidate Sentences**: A list of sentences to evaluate. Each sentence contains a **highlighted word** (e.g., enclosed in `**`) which is the primary anchor (use surrounding words only when the word alone is ambiguous or not interpretable). Here are the candidate sentences:
    -   {candidate_sentence_1}
    -   {candidate_sentence_2}
    -   {candidate_sentence_3}
    -   {candidate_sentence_4}
    -   {candidate_sentence_5}

### **Evaluation Guidelines**
For each highlighted word, determine if the highlighted region of interest is related. First consider the highlighted word itself; only bring in its short local phrase if the word alone is ambiguous.

The relationship can be one of three types:

* **Concrete**: The relationship is **concrete** if the highlighted word (or its immediate phrase) literally describes something visible **inside the red bounding box**. This includes:
    -   Object names and their descriptions (e.g., "a red **Ford** Mustang", "the dog's floppy **ear**")
    -   Visible attributes (e.g., "the building has a **blue** roof", "a tire with a **smooth** texture")
    -   Visible text/characters (e.g., "a sign that reads **STOP** in white letters")

* **Abstract**: The relationship is **abstract** if the highlighted word (or its immediate phrase) describes a concept, function, activity, or quality implied by what is shown in the boxed region, but not literally visible. This includes:
    -   Cultural concepts or qualities (e.g., "the car exudes an aura of **luxury**", "a building with a **traditional** design")
    -   Emotions or feelings (e.g., "the sunset creates a **beautiful** view", "the dark alley feels **scary**")
    -   Implied activities or functions (e.g., "the car is designed for **transportation**", "the engine is responsible for **propulsion**")

* **Global**: The relationship is **global** if the highlighted word (or its immediate phrase) describes something that is literally present somewhere else in the image (**outside the red bounding box**), but not in the highlighted region itself.
    -   (e.g., The red box is on a car's wheel, and the sentence is "the driver is wearing a blue **shirt**". The shirt is visible elsewhere but not in the box).


**Core Logic:**
1.  Mark `interpretable` true if a relationship (concrete, abstract, or global) exists between the boxed region and the highlighted word (optionally its immediate phrase when needed).
2.  The highlighted word must remain the anchor; if only a distant part of the sentence relates and the highlighted word itself does not, mark `interpretable` false.
3. It is okay if a word/phrase is not exactly identical to what is shown visually, but just related. Examples: Image shows a house and word is "castle"/"chateau" (related --> concrete), image shows a car and word is "truck" (related --> concrete), image shows a nice landscape and word is "beautiful" (related --> abstract). The upshot: we are lenient in those cases. 

### **Output Format**
After some reasoning to think first about each word, and possibly next about the few precedding words, return a JSON array with 5 entries, one for each sentence evaluated in order. Each dictionary in the list should have the following structure:
```json
{{
  "id": integer, // The 0-indexed ID of the sentence (0-4)
  "interpretable": boolean, // true if the patch is related to the highlighted word/phrase, otherwise false
  "reasoning": "string", // A brief explanation of why this relationship was determined
  // The following keys are ONLY present if "interpretable" is true
  "relation": "concrete" | "abstract" | "global", // The type of relationship found
  "interpretable_phrase": "string" // The relevant words (or phrase) that explains the relationship. This may be just the highlighted word.
}}
"""

# Sentence-level prompt variant that explicitly includes a cropped region
SENTENCE_LEVEL_PROMPT_WITH_CROP = """You are a visual interpretation expert specializing in connecting contextual textual concepts to specific image regions. Your task is to analyze a list of sentences, each with a highlighted word, and determine how strongly the concept described by the highlighted word (and, if needed, its immediate phrase context) relates to the highlighted region of the image.

### **Inputs**
1.  **Full Image**: A full image with a red bounding box marking the region of interest.
2.  **Cropped Region**: A close-up view of the exact boxed region. Use the crop to read fine details or tiny text; otherwise rely on the full image for context.
3.  **Candidate Sentences**: A list of sentences to evaluate. Each sentence contains a **highlighted word** (e.g., enclosed in `**`) which is the primary anchor (use surrounding words only when the word alone is ambiguous or not interpretable). Here are the candidate sentences:
    -   {candidate_sentence_1}
    -   {candidate_sentence_2}
    -   {candidate_sentence_3}
    -   {candidate_sentence_4}
    -   {candidate_sentence_5}

### **Evaluation Guidelines**
For each highlighted word, determine if the highlighted region of interest is related (even if broadly or abstractly!). First consider the highlighted word itself; only bring in its short local phrase if the word alone is ambiguous.

The relationship can be one of three types:

* **Concrete**: The relationship is **concrete** if the highlighted word (or its immediate phrase) literally describes something visible **inside the red bounding box**. This includes:
    -   Object names and their descriptions (e.g., "a red **Ford** Mustang", "the dog's floppy **ear**")
    -   Visible attributes (e.g., "the building has a **blue** roof", "a tire with a **smooth** texture")
    -   Visible text/characters (e.g., "a sign that reads **STOP** in white letters")

* **Abstract**: The relationship is **abstract** if the highlighted word (or its immediate phrase) describes a concept, function, activity, or quality implied by what is shown in the boxed region, but not literally visible. This includes:
    -   Cultural concepts or qualities (e.g., "the car exudes an aura of **luxury**", "a building with a **traditional** design")
    -   Emotions or feelings (e.g., "the sunset creates a **beautiful** view", "the dark alley feels **scary**")
    -   Implied activities or functions (e.g., "the car is designed for **transportation**", "the engine is responsible for **propulsion**")

* **Global**: The relationship is **global** if the highlighted word (or its immediate phrase) describes something that is literally present somewhere else in the image (**outside the red bounding box**), but not in the highlighted region itself.
    -   (e.g., The red box is on a car's wheel, and the sentence is "the driver is wearing a blue **shirt**". The shirt is visible elsewhere but not in the box).

**Core Logic:**
1.  Mark `interpretable` true if a relationship (concrete, abstract, or global) exists between the boxed region and the highlighted word (optionally its immediate phrase when needed).
2.  The highlighted word must remain the anchor; if only a distant part of the sentence relates and the highlighted word itself does not, mark `interpretable` false.
3. It is okay if a word/phrase is not exactly identical to what is shown visually, but just related. Examples: Image shows a house and word is "castle"/"chateau" (related --> concrete), image shows a car and word is "truck" (related --> concrete), image shows a nice landscape and word is "beautiful" (related --> abstract), "helmet" is conceptually close to "tiara" (related --> abstract). The upshot: we are lenient in those cases and do word associations. And we consider abstract relationships as well.
4. The MOST IMPORTANT RULE: YOU MUST FIRST DISCUSS IN YOUR REASONING IF THE WORD IN ISOLATION COULD BE INTERPRETABLE. LOOK AT IT IN ISOLATION AND IF YOU SEE ANY CONNECTION TO THE REGION OR THE WHOLE IMAGE CONTEXT. ONLY AFTER THAT, YOU CAN LOOK AT THE WORD IN THE CONTEXT OF THE SENTENCE.
5. If the word is not interpretable in isolation, the rest of the sentence can still make the overall judgement interpretable! Importantly, it is okay if not every concept in the sentence is related but only 1-2 concepts are related. Example: Sentence is "beautiful sunset is shining with clouds on the **left**" -- even if there are only clouds and no sunset and no "left", this is definitely interpretable (very important!).

### **Output Format**
After some reasoning to think first about each word, and possibly next about the few precedding words, return a JSON array with 5 entries, one for each sentence evaluated in order. Each dictionary in the list should have the following structure:
{{
  "id": integer, // The 0-indexed ID of the sentence (0-4)
    "reasoning": "string", // Go through the possible ways of being interpretable in order: single word, then sentence (individual concepts in the sentence as a "or function", not a "and function"), and the 3 types of relationships.
  "interpretable": boolean, // true if the patch is related to the highlighted word/phrase, otherwise false
  // The following keys are ONLY present if "interpretable" is true
  "relation": "concrete" | "abstract" | "global", // The type of relationship found
  "interpretable_phrase": "string" // The relevant words (or phrase) that explains the relationship. This may be just the highlighted word. If the word is not interpretable, this is an empty string.
}}
"""
