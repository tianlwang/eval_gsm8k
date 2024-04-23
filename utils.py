import re
from transformers import StoppingCriteria


# Define a stopping condition for generation
class SpecificStringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[self.input_len:]
        
        return any(stop_string in current_text for stop_string in self.stop_strings)


def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None

def extract_ground_truth(text):
    return text.split('####')[-1].strip()