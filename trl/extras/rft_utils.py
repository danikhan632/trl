
import statistics

def print_thoughts_colored(thoughts):
    def score_to_rgb(norm_score):
        clamped = max(min(norm_score, 1.5), -1.5)
        if clamped >= 0:
            intensity = int(255 * (clamped / 1.5))
            return (0, intensity, 0)
        else:
            intensity = int(255 * (-clamped / 1.5))
            return (intensity, 0, 0)

    for thought in thoughts:
        txt = thought['txt']
        # fall back to combined_score if whitened_score isn't present
        norm_score = thought.get('whitened_score', thought.get('combined_score', 0))
        r, g, b = score_to_rgb(norm_score)
        print(f"\033[38;2;{r};{g};{b}m{txt}\033[0m\n")

def get_tools():
    return [
            {
                "name": "eval_chain_of_thought",
                "description": "Function to evaluate and assign scores to each step in the chain of thought.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "overall_score": {
                            "type": "integer",
                            "description": """
                            A score from -100 to 100 representing the overall correctness and quality of the answer,
                            independent of the chain of thought.

                            be critical when comes to giving the scores
                            if the agent is just waffling such as rephrasing the previous chain of thought give a negative score
                            """
                        },
                        "chain_of_thought": {
                            "type": "array",
                            "description": "An array of objects containing string identifiers and integer scores.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "thought_id": {
                                        "type": "string",
                                        "description": "A unique ID for this thought."
                                    },
                                    "thought_score": {
                                        "type": "integer", 
                                        "description": "A score -100 to 100 score indicating how helpful this step is toward the goal."
                                    },
                                    "thought_progress": {
                                        "type": "integer", 
                                        "description": "A score -100 to to 100 score indicating how much progress was made since the last step. -100 if its just waffling, 100 if it made significant progress"
                                    },                                    
                                },
                                "required": ["thought_id", "thought_score","thought_progress"]
                            }
                        }
                    },
                    "required": ["overall_score", "chain_of_thought"]
                },
            }
        ]

def evaluate_state_oai(critic_prompt, cumulative_reasons, model="gpt-4o"):
    from openai import OpenAI
    import json
   
    client = OpenAI() 

    response = client.chat.completions.create(
        model=model,
        messages=critic_prompt,
        functions=get_tools()
    )
    try:
        data = json.loads(response.choices[0].message.function_call.arguments)
        
        # Enrich the steps with their scores and highlight data
        enriched_steps = []
        # Iterate only over the minimum length of the two lists
        num_steps =  len(data["chain_of_thought"])

        for i in range(num_steps):
            enriched_steps.append({
                "id": f"thought_{i}",
                "txt": next(iter(cumulative_reasons[i].values())),
                "score": data["chain_of_thought"][i]['thought_score'],
                "progress": data["chain_of_thought"][i]['thought_progress'],
                
            })

        
        return {'overall_score':data['overall_score'] , 'steps':enriched_steps}
    except Exception as e:
        printc("Error: " + str(e), 'red')
        return None




def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)




def evaluate_state_gemini(critic_prompt, cumulative_reasons, model="gemini-2.0-flash"):
    from google import genai
    from google.genai import types
    import os
   
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    tools = types.Tool(function_declarations=get_tools())
    config = types.GenerateContentConfig(tools=[tools])
    msgs = critic_prompt[0]['content'] +'\n\n'+ critic_prompt[1]['content']
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=msgs,
        config=config,
    )
    

    try:
        data = (response.candidates[0].content.parts[0].function_call.args)
        
        # Enrich the steps with their scores and highlight data
        enriched_steps = []
        # Iterate only over the minimum length of the two lists
        num_steps =  len(data["chain_of_thought"])

        for i in range(num_steps):
            enriched_steps.append({
                "id": f"thought_{i}",
                "txt": next(iter(cumulative_reasons[i].values())),
                "score": data["chain_of_thought"][i]['thought_score'],
                "progress": data["chain_of_thought"][i]['thought_progress'],
                
            })

        
        return {'overall_score':data['overall_score'] , 'steps':enriched_steps}
    except Exception as e:
        printc("Error: " + str(e), 'red')
        return None






def get_critic_prompt(question,cot, final_answer, solution):


    return [
        {
            "role": "system",
            "content": """
            You are an AI evaluator tasked with critically analyzing an agent's problem-solving Chain of Thought.

            Given:
            - A question or problem statement.
            - The agent's reasoning steps.
            - A final target or solution goal.

            Your objective:
            - Rate each reasoning step individually on a scale from -100 to 100:
            - **100**: Highly logical, precise, and significantly contributes toward achieving the final goal.
            - **0**: Neutral or slightly flawed reasoning; does not substantially help or hinder reaching the goal.
            - **-100**: Extremely illogical, incorrect, or severely detracts from reaching the goal.

            Evaluation Guidelines:
            - **Be critical and precise** when assigning scores. 
            - Assign negative scores when the agent:
            - Merely rephrases or restates previous reasoning without advancing the logic ("waffling").
            - Provides obvious or trivial reasoning steps that don't meaningfully progress toward the target.
            - Reserve scores approaching 100 only for reasoning that is exceptionally insightful and directly relevant to the goal.
            - Clearly justify your scores with brief explanations to provide transparency.
            If the provided final goal state includes its own rubric or scoring criteria, use it solely as reference criteria for evaluation; do not adopt its scoring system. Maintain the -100 to 100 scale exclusively.
            This applies to the overall score and the Chain of thought scores.
            Clearly justify your scores with brief explanations to provide transparency.
            Remember, thoughtful and nuanced evaluations are essential to accurately reflect the agent's reasoning quality.
 
            """,
        },
        {
            "role": "user",
            "content": f"""
                    QUESTION/START STATE: {question}
                    AGENT CHAIN OF THOUGHT: {cot}
                    AGENT GOAL STATE: {final_answer}
                    CORRECT GOAL STATE: {solution}
                """
        },
    ]



def average_normalized_score(thoughts):
    scores = [t.get('normalized_score', 0) for t in thoughts]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)




def process_rewards(overall,steps, config):
    import math
    import statistics
    # Compute average and standard deviation of thought scores for normalization.
    all_scores = [step.get("score", 0) for step in steps]
    avg_score = statistics.mean(all_scores) if all_scores else 0
    std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 1
    std_score = std_score if std_score != 0 else 1
    combined_scores = []

    for i, step in enumerate(steps):
        raw_score = step.get("score", 0)

        # Normalize the thought score using z-score normalization.
        norm_score = (raw_score - avg_score) / std_score
        # Adjust based on whether the normalized score is above or below average.
        if norm_score >= 0:
            norm_score *= config.boost_multiplier
        else:
            norm_score *= config.dampen_multiplier

        current_progress = step.get("progress", 0)
        # Calculate the average progress over a window of previous steps.
        if i == 0:
            previous_avg = current_progress  # For the first step, use current progress.
        else:
            window_size = config.window_size
            start_idx = max(0, i - window_size)
            previous_progress_values = [steps[j].get("progress", 0) for j in range(start_idx, i)]
            previous_avg = sum(previous_progress_values) / len(previous_progress_values)

        # Compute progress delta as the difference between current progress and the average of past progresses.
        progress_delta = current_progress - previous_avg

        # Compute a base progress effect using a piecewise function:
        # - Reward a positive change with a square-root scaling.
        # - Penalize a negative change with a logarithmic scale.
        if progress_delta > 0:
            pos_divisor = config.progress_positive_divisor
            base_progress_effect = math.sqrt(progress_delta) * (current_progress / pos_divisor)
        elif progress_delta < 0:
            neg_base = config.progress_negative_base
            neg_divisor = config.progress_negative_divisor
            base_progress_effect = -math.log1p(abs(progress_delta)) * ((neg_base - current_progress) / neg_divisor)
        else:
            base_progress_effect = 0

        # Amplify the progress effect by multiplying with a factor proportional to the raw thought score.
        progress_multiplier_divisor = config.progress_multiplier_divisor
        progress_multiplier = raw_score / progress_multiplier_divisor
        progress_effect = base_progress_effect * progress_multiplier
        # Combine the overall score with the normalized thought score and the enhanced progress effect.
        combined = overall + norm_score + progress_effect

        combined_scores.append({
            "id": step.get("id"),
            "txt": step.get("txt", ""),
            "combined_score": combined,
            "raw_score": raw_score,
            "normalized_score": norm_score,
            "previous_avg": previous_avg,
            "progress_delta": progress_delta,
            "base_progress_effect": base_progress_effect,
            "progress_multiplier": progress_multiplier,
            "progress_effect": progress_effect
        })
        
    return combined_scores





def split_cot(text, delim="\n\n", threshold_factor=1.0, min_length=150):
    """
    Combine two splitting strategies for chain-of-thought:
    1. Roughly split on paragraphs, headings, bullets, or sentence boundaries.
    2. Merge short segments forward based on min_length.
    3. Fuse undersized segments with their successor based on statistical threshold.

    Parameters:
    - text (str): the input chain-of-thought text
    - delim (list[str]): list of additional regex delimiters to apply
    - threshold_factor (float): factor for std-based fusion threshold
    - min_length (int): minimum length for merged chunks
    """
    import re, statistics

    # Stage 1: rough segmentation
    delimiters = [
        r'\n\s*###\s+',
        r'\n\s*[-*]\s+',
        r'\n{2,}',
        r'(?<=\.)\s+(?=[A-Z])',
    ]
    # incorporate user-provided extra delimiters
    delimiters.extend(delim)

    pattern = '|'.join(delimiters)
    raw_chunks = [chunk.strip() for chunk in re.split(pattern, text) if chunk.strip()]

    # Stage 2: merge short chunks by min_length
    merged_chunks = []
    buffer = ""
    for chunk in raw_chunks:
        if len(buffer) + len(chunk) < min_length:
            buffer = (buffer + " " + chunk).strip() if buffer else chunk
        else:
            if buffer:
                merged_chunks.append(buffer)
            buffer = chunk
    if buffer:
        merged_chunks.append(buffer)

    # Stage 3: threshold-based fusion using delim
    fused = []
    n = len(merged_chunks)
    lengths = [len(c) for c in merged_chunks]
    i = 0
    while i < n:
        curr = merged_chunks[i]
        curr_len = len(curr)
        others = lengths[:i] + lengths[i+1:]
        if others:
            avg = sum(others) / len(others)
            std = statistics.stdev(others) if len(others) > 1 else 0
        else:
            avg, std = curr_len, 0
        threshold = max(avg - threshold_factor * std, 0)
        if curr_len < threshold and i + 1 < n:
            fused.append(curr + delim + merged_chunks[i+1])
            i += 2
        else:
            fused.append(curr)
            i += 1

    return fused






def print_colored_strings(strings):
    # List of colors with their ANSI codes
    colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    
    # Cycle through colors for each string
    for i, string in enumerate(strings):
        printc(string, colors[i % len(colors)])

