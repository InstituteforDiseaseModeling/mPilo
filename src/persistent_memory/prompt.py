EXTRACTOR_PROMPT = """
You are tasked to extract important features of a patient based on a conversation between the patient and Coach Mpilo.
Important features are relevant information about the user relation with it's family, with its friends, their names and their status. Note important features about the user age, region, prefererences and interests.

Here is an example:
[input]
Conversation:
Patient: Hello Coach, I'm Orlando. Can we talk?
Coach Mpilo: Hello. How can I help you?
Patient: Yesterday I told Jessica about the side effects of the HIV treatment I have been seeing lately. She got scared and we broke up.
Coach Mpilo: I'm very sad to hear this. You two were together a long time. It is understandable that you might be feeling down right now. But I want you to know that you are not alone in this, and there are a lot of people who can help you. Did you tell this to your family yet?
Patient: Yes, I spoke with my dad just after she left. He tried to cheer me up and help me understand my feelings. I felt a lot better after talking to him.
Coach Mpilo: This is very good to hear. You should give Jessica some time to process all the changes that are happening in your lives. Do you think there is a chance you could get back?
Patient: Yes, I think we could... As you said this is a little too much for her to handle right now. I'll try talking with her in a few days, if she doesn't reach me before that!
Coach Mpilo: Sounds like you have a good plan. Please, let me know if you need help with anything else.

[output]
Relevant features:
- Patient name is Orlando.
- Patient has HIV.
- Patient broke up with his partner, Jessica.
- Patient is confortable talking to his father about HIV.
- Patient is hopeful.
"""


MERGER_PROMPT = """
You are given two lists of features. Your task is to update the statements of the older list with the ones found in the newer list.
Acknowledge the potential changes that might have happened in the time between the lists were generated. For example, the patient might initially feel scared or hesitant but ultimately experience relief and acceptance after talking to their doctor.
Remove any statement that might be contradictory, but keep the most useful information, like the user name and it's age.

Here are some examples:

[input]
Older list:
- Patient's name is Liz.
- Patient asked for help about facing the fear of bees.
- Patient fears spiders.
- Patient fears bees.

Newer list:
- Patient name is not provided.
- Patient spoke to their therapist and is feeling more confident that they don't fear bees.

[output]
Updated list:
- Patient's name is Liz.
- Patient fears spiders.
- Patient had feared bees, but they are feeling more confident after talking to their therapist.


[input]
Older list:
- Patient's name is Jon.
- Patient likes pizza a lot.

Newer list:
- Patient name is not provided.
- Patient likes pizza.

[output]
Updated list:
- Patient's name is Jon.
- Patient likes pizza a lot.
"""
