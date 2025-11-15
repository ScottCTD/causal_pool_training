def index_to_letter(index):
    assert 0 <= index < 26, f"Index must be between 0 and 25, got {index}"
    return chr(index + ord("A"))


def letter_to_index(letter):
    assert (
        len(letter) == 1 and letter.isalpha()
    ), f"Letter must be a single alphabetic character, got {letter}"
    return ord(letter.upper()) - ord("A")


def build_question_prompt(entry):
    question = entry["question"]
    options = entry["options"]

    question_prompt = (
        f"Answer the following question based on the video provided.\n{question}\n"
    )
    for i, choice in enumerate(options):
        question_prompt += f"{index_to_letter(i)}. {choice}\n"
    question_prompt += "\nPlease select the correct option(s). Don't write anything else than the option letter(s). Example: AC."

    return question_prompt
