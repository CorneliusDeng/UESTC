from EntityExtractor import EntityExtractor
from SearchAnswer import AnswerSearching
import warnings
warnings.filterwarnings('ignore')


class SIRI:
    def __init__(self):
        self.extractor = EntityExtractor()
        self.searcher = AnswerSearching()

    def qa_main(self, input_str):
        noanswer = "对不起，您的问题我不知道，我今后会努力改进的。"
        entities = self.extractor.extractor(input_str)
        if not entities:
            return noanswer
        sqls = self.searcher.question_parser(entities)
        final_answer = self.searcher.searching(sqls)
        if not final_answer:
            return noanswer
        else:
            return '\n'.join(final_answer)


if __name__ == "__main__":
    handler = SIRI()
    while True:
        question = input("User：")
        if not question:
            break
        answer = handler.qa_main(question)
        print("Assistant：", answer)
        print("-"*50)
