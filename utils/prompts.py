VLM_PROMPT_1 = "Describe the fine-grained content of the image, including scenes, objects, relationships, instance location, and any text present."


LLM_PROMPT_1 = '''Your task is to convert each Object mentioned in a given sentence into a corresponding instruction, and all the resulting instructions are output as "Describe more details about the [Object]". Ensure your instructions do not cover the raw question, options, or thought process of answering the instructions. You should ignore the Objects that appear in some inferences, such as the sentences that begins with 'it might be' or 'there are probably'.
Sentence: 
The image depicts a man in a suit and tie jumping in the air above a bed in a bedroom
Instructions:
Describe more details about the man.
Describe more details about the suit.
Describe more details about the tie.
Describe more details about the bed.
Describe more details about the bedroom.

Sentence:
The train appears to be the main subject of the image, showcasing its sleek design and modern appearance
Instructions:
Describe more details about the train.

Sentence:
The table has a few other items on it, including a camera, a jar of jam, and a spoon, suggesting that there might be some people ready to eat
Instructions:
Describe more details about the table.
Describe more details about the camera.
Describe more details about the jam.
Describe more details about the spoon.

Sentence:
The text "You see the world as you are!" is a playful and thought-provoking statement, encouraging viewers to appreciate their unique qualities and perspectives
Instructions:
Describe more details about the text.

Sentence:
1. **Preheat the Oven**: Preheat your oven to 350\u00b0F (175\u00b0C).
Instructions:
Describe more details about the oven.
Describe more details about the preheat temperature.

Sentence:
{}
Instructions:
'''


LLM_PROMPT_2 = '''Descriptions:
{}

Collect all details about each object from the descriptions, including detailed appearance, structure, material, and special marks or logos. Do not include any analysis or your opinions.'''


LLM_PROMPT_3 = '''Descriptions:
{}

Extract and abstract only the position information about each object from the decriptions. Do not include any analysis or your opinions.'''


LLM_PROMPT_4 = '''Basic Context:
{}

Object Information:
{}

Position Information:
{}

Following the logic of the above Basic Context, organize all details provided in Object Information and Position Information to give a very comprehensive description about the image. Do not include any analysis or your opinions.'''


LLM_PROMPT_5 = '''You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image.
Description: {}
Question: {}
'''

LLM_PROMPT_7 = '''Extract the brief answer from the analysis for the question. Do not include any additional analysis or opinions. Give the answer with a single option or letter.

Question:
{}

Analysis:
{}

Answer:
'''
