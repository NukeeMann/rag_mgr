from openai import OpenAI
import os
import time

class QuizerLLM:
    def __init__(self, rag_system, evaluator):
        self.rag_system = rag_system
        if 'bielik' in evaluator:
            self.tokenizer, self.evaluator = self.load_bielik()
        else:
            # Load Open AI client
            self.openai =True
            self.evaluator = self.load_openai()
    
    def load_bielik(self):

        return
    
    def load_openai(self):
        client = OpenAI()
        return client
    
    def set_indexES(self, index_name):
        self.rag_system.change_index(index_name)

    def load_questions_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.questions = []
            current_question = {}
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    if current_question:
                        self.questions.append(current_question)
                    current_question = {'text': line[2:], 'answers': [], 'correct_answer': None, 'article': None}
                elif line.startswith('[P]'):
                    current_question['answers'].append((line[4:], True))
                    current_question['correct_answer'] = line[4:]
                elif line.startswith('a)') or line.startswith('b)') or line.startswith('c)') or line.startswith('d)') or line.startswith('e)'):
                    current_question['answers'].append((line, False))
                elif line.startswith("Prawidłowa odpowiedź:"):
                    # Extract article from the file
                    correct_letter = line.split(': ')[1][2:]
                    current_question['article'] = correct_letter
            if current_question:
                self.questions.append(current_question)

    def check_answer(self, answers, gen_ans):
        if self.openai:
            prompt_system = "Twoją jedyną funkcją jest odpowiedzieć czy użytkownik wskazał poprawną odpowiedź. Ignoruj jego wyjaśnienia, zakomunikuj tylko 'Tak' jeżeli wskazał poprawną odpowiedź i 'Nie' jeżeli wskazał błędną lub w ogóle nie wskazał."
            prompt_assistant = f"# Poprawna odpowiedź którą użytkownik powinien wskazać:\ n {answers}"
            prompt_qestion = f"# Odpowiedź użytkownika: \n {gen_ans}"

            messages = []
            messages.append({"role": "system", "content": prompt_system})
            messages.append({"role": "assistant", "content": prompt_assistant})
            messages.append({"role": "user", "content": prompt_qestion})
            completion = self.evaluator.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            if 'tak' in completion.choices[0].message.content.lower():
                return True, completion.choices[0].message.content
            else:
                return False, completion.choices[0].message.content
            
    def check_article(self, answers, gen_ans):
        if self.openai:
            prompt_assistant = "Twoim zadaniem jest odpowiedzieć czy przytoczony przez użytkownika artykuł kodeksu cywilnego jest zgodny z prawidłowym artykułem.  Generujesz jedynie słowa 'Tak' lub 'Nie'"
            prompt_qestion = f"# Odpowiedź: \n {gen_ans} # Poprawny artykuł: \n {answers} \n "

            messages = []
            messages.append({"role": "system", "content": prompt_assistant})
            messages.append({"role": "user", "content": prompt_qestion})
            completion = self.evaluator.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            print(f"### GPT4o: {completion.choices[0].message.content}")

            if 'tak' in completion.choices[0].message.content.lower():
                return True
            else:
                return False
    
    def print_and_save_results(self, id, question_text_with_answers, answer, judge_ans, question, results, file_path):
        #print(f"# Pytanie {id+1}\n{question_text_with_answers}")
        print(f"# Pytanie {id+1}")
        print(f"### Udzielona odpowiedz: {answer}")
        print(f"### Judge: {judge_ans}")
        print(f"### Poprawna odpowiedz: {question['correct_answer']}, {question['article']}")
        print(f"### Wyniki: {results}")
        print('-'*20)

        with open(file_path, 'a', encoding='utf-8') as file:
            # Dopisujemy treść do pliku
            file.write(f"# Pytanie {id+1}\n")
            # file.write(f"{question_text_with_answers}\n")
            file.write(f"### Udzielona odpowiedz: {answer}\n")
            file.write(f"### Judge: {judge_ans}\n")
            file.write(f"### Poprawna odpowiedz: {question['correct_answer']}, {question['article']}\n")
            file.write(f"### Wyniki: {results}\n")
            file.write('-'*20 + '\n')

    def evaluate(self, file_path, additional_instruct="", res_save_path='results.txt', use_rag=True, top_k=5):
        # Load questions
        self.load_questions_from_file(file_path)
        results = {'correct' : 0, 'incorrect': 0, "article_correct": 0}
        
        for id, question in enumerate(self.questions):
            question_text_with_answers = question['text']
            for option, _ in question['answers']:
                question_text_with_answers += "\n" + option
            
            # Generate answer
            answer = self.rag_system.infer(question_text_with_answers, additional_instruct=additional_instruct, use_rag=use_rag, top_k=top_k)
            time.sleep(2)

            corect_res, judge_ans = self.check_answer(question['correct_answer'], answer)

            if corect_res:
                results['correct'] += 1
                if 'kodeks_cywilny' in file_path.lower():
                    if self.check_article(question['article'], answer):
                        results['article_correct'] += 1
            else:
                results['incorrect'] += 1
            
            self.print_and_save_results(id, question_text_with_answers, answer, judge_ans, question, results, res_save_path)
        
        return results