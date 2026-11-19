import math
import pandas as pd
import numpy as np
import re
import ast


def find_non_None(ref_list):
    """Find the index of the items that are not a None within a list.

    Parameters
    ----------
    ref_list : list
        List to search for valid items.

    Returns
    -------
    list
        Returns a list of indexes of valid items within the list.

    """
    return [index for index, item in enumerate(ref_list) if item is not None]


def freq_prob(neutral_weight):
    """Return a data.frame with segment assignation probability based on frequency, adjusted by the neutral_weight.

    Parameters
    ----------
    neutral_weight : float
        Value to be used to penalize known classes and to increase neutral assignation probability.

    Returns
    -------
    data.frame
        Returns a data.frame with segments and probabilities.

    """
    segments = ["Teal", "Rose", "Green", "Blue", "Gray"]
    starting_probs = [0.23, 0.24, 0.15, 0.22, 0.16]

    adjusted_probs = [prob * (1 - neutral_weight) for prob in starting_probs]

    segments.append("Opaque")
    adjusted_probs.append(neutral_weight)
    prob_df = pd.DataFrame({"segment": segments, "prob": adjusted_probs, "evidence": 1})

    return prob_df


def cluster_prob(centroids, patient_values):
    """Given a set of centroids and new values, find the probability of assignation to each segment.

    Parameters
    ----------
    centroids : pandas data.frame
        A S x (F+1) data.frame containing the information of each centroid. Where S is the number of segments (rows) and F is the number of features + 1 (columns).
    patient_values : list
        The list of values of the new observation to classify of dimension 1 x F.

    Returns
    -------
    pandas data.frame
        A pandas data.frame of dimension S x 2. Segment names are on the first column and probability of assignation on the second.
    """
    # TODO: verify if the values and the centroids are in the correct object classes (pd.df and list)

    # TODO: add logic to replace non-numerical data with None
    # Print a warning in these cases

    # TODO: add logic to verify if length of input matches the centroids'
    # Throw an error in case of diff lengths

    # TODO: add logic to coerce all values to float

    # Get valid columns
    colnames = [centroids.columns.tolist()[i + 1] for i in find_non_None(patient_values)]

    # Get valid values
    patient_fill = [patient_values[i] for i in find_non_None(patient_values)]
    valid_per = len(patient_fill) / len(patient_values)

    # Convert list to np.array
    pat_np = np.array(patient_fill, dtype=np.float64)

    all_dist = []
    for segment in centroids.iterrows():
        segment_values = segment[1][colnames].to_numpy()
        dist = np.linalg.norm(segment_values - pat_np)
        all_dist.append(dist)

    # Calculate the individual assignation confidence
    seg_conf = [math.exp(-i) for i in all_dist]

    # Penalize confidence with the valid counts percentage
    pen_conf = [i * valid_per for i in seg_conf]
    # Add confidence of misplacing due to lack of info
    pen_conf = pen_conf + [1 - valid_per]

    # Calculate relative prob
    rel_prob = [i / sum(pen_conf) for i in pen_conf]

    # Create df to be returned
    ret_centroids = centroids.copy()[["segment"]]
    # Add neutral category
    ret_centroids.loc[len(ret_centroids.index)] = ["Opaque"]
    ret_centroids["prob"] = rel_prob
    ret_centroids["evidence"] = 1
    return ret_centroids


def equal_probs():
    """Return a data.frame with almost equal probs to all segments. The neutral segment has a slightly higher probability.

    Returns
    -------
    pandas data.frame
        A pandas data.frame of.
    """
    segments = ["Teal", "Rose", "Green", "Blue", "Gray", "Opaque"]
    equal_probs = [0.166, 0.166, 0.166, 0.166, 0.166, 0.17]
    prob_df = pd.DataFrame({"segment": segments, "prob": equal_probs, "evidence": 1})
    return prob_df


def chat_history_prob(conversation_buffer, llm_classifier, neutral_weight=0):
    """Given a converastion buffer and a LLM prompted to classify it, find the probability of assignation to each segment.

    Parameters
    ----------
    conversation_buffer : ConversationBufferMemory
        A Langchain memory conversation buffer with the messages to be used in the classification.
    llm_classifier : Runnable
        A Langchain runnable with an LLM prompted to classify the given conversation.
    neutral_weight : float
        In the case that the LLM fails to classify the given conversation, this argument is passed to the frequentist probability.

    Returns
    -------
    pandas data.frame
        A pandas data.frame with the probabilities for each segment.
    """
    # Extract chat history from memory
    chat_dict = conversation_buffer.load_memory_variables({})
    if "chat_history" in chat_dict:
        chat_history = chat_dict["chat_history"]
    else:
        # No chat history found
        chat_history = ""

    # Count the number of human interactions with the agent
    evidence = len(re.findall("\nHuman:", chat_history))

    if evidence > 1:
        # Use llm to try and classify based on the whole chat history
        classifier_response = llm_classifier.invoke({"input": chat_history})
        try:
            classifier_text = classifier_response.content
            cleaned_text = classifier_text[
                classifier_text.find("{") : classifier_text.rfind("}") + 1
            ]  # Extract the dictionary based on curly brackets
            resp_dict = ast.literal_eval(cleaned_text)
            probs = pd.DataFrame.from_dict(resp_dict, orient="index", columns=["prob"])
            probs.index.name = "segment"
            probs.reset_index(inplace=True)
            probs["evidence"] = evidence
        except Exception as ex:
            template = "An exception of type {0} occurred when assessing the segment class. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            resp_dict = None  # LLM failed to provide the output in the correct format
            probs = equal_probs()
    else:
        # If there are no user input messages, return the default probability based on class frequency
        probs = freq_prob(neutral_weight=neutral_weight)

    return probs


def new_input_prob(conversation_buffer, new_input, llm_classifier):
    """Given a converastion buffer, a new user input and a LLM prompted to classify it, find the probability of assignation to each segment.

    Parameters
    ----------
    conversation_buffer : ConversationBufferMemory
        A Langchain memory conversation buffer with the messages to be used in the classification.
    new_input : str
        A string containing the new user input. This value is appended to the history extracted from the conversation_buffer.
    llm_classifier : Runnable
        A Langchain runnable with an LLM prompted to classify the given conversation.

    Returns
    -------
    pandas data.frame
        A pandas data.frame with the probabilities for each segment.
    """
    # Extract chat history from memory
    chat_dict = conversation_buffer.load_memory_variables({})
    if "chat_history" in chat_dict:
        chat_history = chat_dict["chat_history"]
        sep = "\nHuman: "
    else:
        # No chat history found
        chat_history = ""
        sep = ""

    new_history = chat_history + sep + new_input
    # Use llm to try and classify based on the whole chat history + new input
    classifier_response = llm_classifier.invoke({"input": new_history})

    try:
        if type(classifier_response) is str:
            classifier_text = classifier_response
        else:
            classifier_text = classifier_response.content

        cleaned_text = classifier_text[
            classifier_text.find("{") : classifier_text.rfind("}") + 1
        ]  # Extract the dictionary based on curly brackets
        resp_dict = ast.literal_eval(cleaned_text)
        probs = pd.DataFrame.from_dict(resp_dict, orient="index", columns=["prob"])
        probs.index.name = "segment"
        probs.reset_index(inplace=True)
        probs["evidence"] = 1
        # Evidence = 1 -> only one new input, the rest of the history is just for context

    except Exception as ex:
        template = "An exception of type {0} occurred when assessing the segment class. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        resp_dict = None  # LLM failed to provide the output in the correct format
        probs = equal_probs()

    return probs


def highest_prob(segment_probs):
    """Given a data.frame of segments and probabilities, return the name of the segment with highest probability of assignation.

    Parameters
    ----------
    segment_probs : pandas data.frame
        A pandas data.frame of dimension S x 2. Segment names are on the first column and probability of assignation on the second.

    Returns
    -------
    str
        A string containing the segment with the highest probability.
    """
    seg_df = segment_probs["segment"][segment_probs["prob"] == max(segment_probs["prob"])]
    return seg_df.iloc[0]


def highest_prob_above_threshold(segment_probs, null_class, threshold):
    """Given a data.frame of segments and probabilities, return the name of the segment with highest probability of assignation that is above the given threshold and ignoring the null_class.

    Parameters
    ----------
    segment_probs : data.frame
        A pandas data.frame with segments and probabilities.
    null_class : str
        A string with the name of the null class.
    threshold : float
        A float between 0 and 1 with the minimum probability for a class to be considered.

    Returns
    -------
    str
        A string containing the chosen segment.
    """
    segment_probs_valid = segment_probs[~segment_probs["segment"].str.contains(null_class)]

    if all([probs < threshold for probs in segment_probs_valid.prob]):
        segment = null_class
    else:
        segment = highest_prob(segment_probs)
    return segment


def adjust_probs(priori_probs, posteriori_probs):
    """Given two data.frames of segments, probabilities and evidence, for priori and posteriori combine them and return an adjusted version.

    Parameters
    ----------
    priori_probs : data.frame
        A pandas data.frame with priori segments, probabilities and evidence.
    posteriori_probs : data.frame
        A pandas data.frame with posteriori segments, probabilities and evidence.

    Returns
    -------
    data.frame
        A data.frame with updated probabilities and evidence for each segment.
    """
    priori_evidence = priori_probs.evidence.max()
    posteriori_evidence = posteriori_probs.new_evidence.max()
    total_evidence = priori_evidence + posteriori_evidence

    priori_weight = priori_evidence / total_evidence
    posteriori_weight = posteriori_evidence / total_evidence

    # Overwrite possible NaN in the data.frames
    priori_probs.prob = priori_probs.prob.fillna(0)
    posteriori_probs.new_prob = posteriori_probs.new_prob.fillna(0)

    combined_probs = priori_probs.merge(posteriori_probs, how="left", on="segment")

    combined_probs["adjusted_prob"] = (
        combined_probs["prob"] * priori_weight + combined_probs["new_prob"] * posteriori_weight
    )
    new_probs = combined_probs.drop(["prob", "new_prob", "evidence", "new_evidence"], axis=1).rename(
        columns={"adjusted_prob": "prob"}
    )
    new_probs["evidence"] = total_evidence

    # Overwrite possible NaN in the data.frame
    new_probs.prob = new_probs.prob.fillna(0)

    return new_probs


CENTROIDS = pd.DataFrame(
    {
        "segment": ["Teal", "Rose", "Green", "Blue", "Gray"],
        "risk": [1, 4, 5, 3, 2],
        "KZN": [0.52, 0.48, 0.59, 0.79, 0.62],
        "age_0_24": [0.45, 0.43, 0.38, 0.29, 0.42],
        "age_25_29": [0.33, 0.39, 0.37, 0.43, 0.36],
        "age_30_100": [0.22, 0.18, 0.25, 0.28, 0.22],
        "community": [5, 3, 2, 1, 5],
        "family": [4, 3, 2, 2, 4],
        "work": [4, 3, 1, 2, 1],
        "friends": [3, 4, 4, 2, 2],
        "relationships": [2, 4, 3, 2, 2],
        "recreation": [1, 4, 4, 1, 1],
    }
)
