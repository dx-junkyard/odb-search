import os
import json
from openai import OpenAI
import logging
from typing import Tuple, Optional, Dict, List



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

    def collect_user_info(self, question: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        ユーザーからの質問を分析し、必要な情報が十分かどうかを判断し、
        不足している場合は追加の質問を生成する。

        Returns:
            Tuple[bool, Optional[str], Optional[str]]:
            - 十分な情報があるかどうか
            - 追加の質問（情報が不足している場合）
            - 収集した情報の要約（十分な情報がある場合）
        """
        try:
            # 情報収集のためのプロンプト
            info_collection_prompt = """
            あなたは行政サービスの案内係です。ユーザーの質問から、以下の情報を収集する必要があります：

            1. ユーザーの属性（例：年齢、職業、家族構成、収入状況など）
            2. 探しているサービスの具体的な内容（例：目的、状況、必要な支援など）

            以下の質問文を分析し、必要な情報が十分かどうかを判断してください。
            情報が不足している場合は、不足している情報を収集するための質問を生成してください。
            十分な情報がある場合は、収集した情報を要約してください。

            出力形式：
            {
                "has_sufficient_info": true/false,
                "additional_question": "追加の質問（情報が不足している場合）",
                "info_summary": "収集した情報の要約（十分な情報がある場合）"
            }
            """

            prompt = f"{info_collection_prompt}\n\nユーザーの質問：{question}"
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

            try:
                # 応答を直接JSONとして解析
                result = json.loads(response)
                return (
                    result.get('has_sufficient_info', False),
                    result.get('additional_question'),
                    result.get('info_summary')
                )

            except json.JSONDecodeError as e:
                logging.error(f"JSONのパースに失敗しました: {e}")
                logging.error(f"問題のあるJSON文字列: {response}")
                return False, "申し訳ありません。情報の分析に失敗しました。もう一度質問を入力してください。", None

        except Exception as e:
            logging.error(f"情報収集処理中にエラーが発生しました: {str(e)}")
            return False, "申し訳ありません。システムエラーが発生しました。", None

    def generate_labels_from_info(self, info_summary: str) -> Optional[Dict[str, List[str]]]:
        """
        収集した情報から対象者ラベルとサービスラベルを生成する。

        Args:
            info_summary: 収集した情報の要約

        Returns:
            Optional[Dict[str, List[str]]]: 生成されたラベル（対象者ラベルとサービスラベルの辞書）
        """
        try:
            # ラベル生成のためのプロンプト
            label_generation_prompt = """
            以下の情報を分析し、対象者ラベルとサービスラベルを生成してください。

            対象者ラベルは、この情報に関連する可能性のある対象者カテゴリを示します。
            サービスラベルは、この情報に関連する可能性のある行政サービスの種類を示します。

            出力形式：
            {
                "対象者ラベル": ["ラベル1", "ラベル2", ...],
                "サービスラベル": ["ラベル1", "ラベル2", ...]
            }
            """

            prompt = f"{label_generation_prompt}\n\n収集した情報：{info_summary}"
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

            try:
                # 応答を直接JSONとして解析
                return json.loads(response)

            except json.JSONDecodeError as e:
                logging.error(f"JSONのパースに失敗しました: {e}")
                logging.error(f"問題のあるJSON文字列: {response}")
                return None

        except Exception as e:
            logging.error(f"ラベル生成中にエラーが発生しました: {str(e)}")
            return None

    def generate_json_from_question(self, question: str) -> Optional[Dict[str, List[str]]]:
        """
        ユーザーの質問から必要な情報を収集し、ラベルを生成する。

        Args:
            question: ユーザーからの質問

        Returns:
            Optional[Dict[str, List[str]]]: 生成されたラベル（対象者ラベルとサービスラベルの辞書）
        """
        # 情報収集の試行
        has_sufficient_info, additional_question, info_summary = self.collect_user_info(question)

        if not has_sufficient_info:
            # 情報が不足している場合は、追加の質問を返す
            return {
                "needs_more_info": True,
                "additional_question": additional_question
            }

        # 十分な情報がある場合は、ラベルを生成
        labels = self.generate_labels_from_info(info_summary)
        if labels:
            labels["needs_more_info"] = False
            return labels

        return None


