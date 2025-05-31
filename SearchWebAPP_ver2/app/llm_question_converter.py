import os
import json
from openai import OpenAI
import logging



class LLMQuestionConverter:
    def __init__(self, llm_url, llm_api_key, llm_model, llm_prompt_file):
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_prompt = self.load_llm_prompt(llm_prompt_file)
        self.client = OpenAI(
            base_url=self.llm_url,
            api_key=self.llm_api_key,
        )
        #self.client = OpenAI()

    def load_llm_prompt(self, llm_prompt_file):
        try:
            with open(llm_prompt_file, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            logging.error(f"指定されたファイル '{llm_prompt_file}' が見つかりません。")
            return ""
        except Exception as e:
            logging.error(f"エラーが発生しました: {e}")
            return ""

    def convert_html_to_json(self, question):
        try:
            # HTMLの内容をLLMに送信
            prompt = f"{self.llm_prompt}\n{question}"
            chat_completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ]
            )
            response = chat_completion.choices[0].message.content
            
            # JSON配列を抽出
            try:
                # ```json と ``` で囲まれた部分を探す
                json_start = response.find('```json')
                if json_start != -1:
                    json_end = response.find('```', json_start + 6)
                    if json_end != -1:
                        json_str = response[json_start + 7:json_end].strip()
                        try:
                            # JSONとしてパース
                            json_data = json.loads(json_str)
                            if isinstance(json_data, list):
                                return json_data
                            else:
                                # 単一のオブジェクトの場合は配列に変換
                                return [json_data]
                        except json.JSONDecodeError as e:
                            logging.error(f"JSONのパースに失敗しました: {e}")
                            logging.error(f"問題のあるJSON文字列: {json_str}")
                            return None
                
                logging.error(f"有効なJSON配列が見つかりませんでした: {response}")
                return None
                
            except Exception as e:
                logging.error(f"JSONの抽出に失敗しました: {str(e)}")
                return None
                
        except Exception as e:
            logging.error(f"LLMの呼び出しに失敗しました: {str(e)}")
            return None


