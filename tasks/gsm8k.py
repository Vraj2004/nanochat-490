"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from tasks.common import Task, load_hub_dataset


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_hub_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


# -----------------------------------------------------------------------------
# A4 Part 4: shaped-reward RL environments. Same problems and evaluation as GSM8K,
# only the training reward changes. Both bonuses are capped below 1 (finish 0.1,
# partial at most 8 x 0.1), so a correct answer always out-scores a wrong one.

CALC_OUTPUT_RE = re.compile(r"<\|output_start\|>(.*?)<\|output_end\|>", re.S)
NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")

def to_number(text):
    """ Parse a number for numeric comparison ("20.0" == "20", "1,000" == "1000"), None if not a number. """
    try:
        return round(float(text.replace(",", "")), 6)
    except ValueError:
        return None

class GSM8KShaped(GSM8K):
    """
    GSM8K with the correctness reward plus optional shaping terms:
    - finish: +finish_bonus if the final "#### N" is the result of one of the model's own calculator calls
    - partial: +partial_bonus for each intermediate result of the reference solution (calculator results
      other than the final answer, and not just a number from the question) that the model also computes
    """

    def __init__(self, subset, split, finish=False, partial=False, finish_bonus=0.1, partial_bonus=0.1, **kwargs):
        super().__init__(subset, split, **kwargs)
        self.finish, self.partial = finish, partial
        self.finish_bonus, self.partial_bonus = finish_bonus, partial_bonus

    def reward_terms(self, conversation, assistant_response):
        """ Returns (correct, finish, num_matched_intermediates) for logging and the reward. """
        correct = self.evaluate(conversation, assistant_response)
        calc_outputs = {to_number(x) for x in CALC_OUTPUT_RE.findall(assistant_response)} - {None}
        pred = extract_answer(assistant_response)
        finish = int(pred is not None and to_number(pred) in calc_outputs)
        parts = conversation['messages'][-1]['content']
        final = to_number(extract_answer(parts[-1]['text']))
        question_numbers = {to_number(x) for x in NUMBER_RE.findall(conversation['messages'][0]['content'])}
        intermediates = {to_number(p['text']) for p in parts if p['type'] == 'python_output'}
        intermediates -= question_numbers | {final, None}
        matched = len(intermediates & calc_outputs)
        return correct, finish, matched

    def reward(self, conversation, assistant_response):
        correct, finish, matched = self.reward_terms(conversation, assistant_response)
        reward = float(correct)
        if self.finish:
            reward += self.finish_bonus * finish
        if self.partial:
            reward += self.partial_bonus * matched
        return reward
