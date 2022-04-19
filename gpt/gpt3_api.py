
import os
import openai
import configurations as cfg


def GPT_Completion(text: str) -> str:
    openai.api_key = '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'

    response = openai.Completion.create(engine=cfg.gpt_engine,
                                        prompt=text,
                                        temperature=0.6,
                                        top_p=1,
                                        max_tokens=20,
                                        frequency_penalty=0,
                                        presence_penalty=0)

    answer = response.choices[0].text
    answer = answer.strip()

    return answer
