from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import os

class GTTCQA:
    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad', save_dir='./model'):
        self.save_dir = save_dir
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)

        # Define the context here
        self.context = """
The Government Tool Room & Training Centre (GTTC) has centers in Hubli and Belagavi, Karnataka.

Hubli GTTC offers specialized training in tool and die making, precision manufacturing, and related technical skills. It provides diploma and certificate courses such as CNC machining, CAD/CAM design, and advanced manufacturing techniques. Various programs include the COE PTC Hubli and COE Autodesk, with free and paid courses under the CMKKY, AITT SCP, and AITT TSP schemes. Paid courses include CNC programming, AutoCAD, AI, machine learning, and Python. Internship opportunities are available for both diploma and engineering students. The principal is Mr. Maruthi Bhajantri. For more details, contact 0836 2333159.

Belagavi GTTC offers a diverse range of educational and training programs. It provides diploma courses in Tool and Die Making and Precision Manufacturing, with a 3-year study period and 1 year of practical training, totaling 4 years, and a fee of ₹27,500. Eligibility requires passing the 10th standard. The center also offers long-term vocational training in Tool Maker, Tool Room Machinist, and Tool and Die Technician, with durations of 1 to 2 years. Paid courses include CNC Programming & Operations, Mastercam, AutoCAD, Creo (Pro-E), SolidWorks, NX CAD, CATIA, and Autodesk Fusion 360, with fees ranging from ₹2,000 to ₹5,000. Internships are available for BE and Diploma students, and SAP courses are offered with a fee structure of ₹8,008 for certification and ₹7,000 for assessment. COE PTC and COE Smart City programs offer courses in SAP, 3D Printing, Robotics, and more, with a fee of ₹4,000. For more information, contact 0831-2950611 or email gttc_bgm@yahoo.com.
"""
        self.full_form = "Government Tool Room & Training Centre"
        self.location_info = {
            "Hubli": "Gttc Hubli B-467 To B-474, Industrial Estate, Gokul Road, Hubli, Karnataka",
            "Belagavi": "Industrial Estate, Udyambag, Belgaum 590008"
        }

        # Attempt to load the model if it exists
        self.load_model()

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Save model state_dict manually
        model_path = os.path.join(self.save_dir, 'pytorch_model.bin')
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(self.save_dir)

        print(f"Model saved to {self.save_dir}")

    def load_model(self):
        if os.path.exists(self.save_dir):
            # Load model with the same configuration
            self.model = BertForQuestionAnswering.from_pretrained(self.save_dir)
            self.tokenizer = BertTokenizer.from_pretrained(self.save_dir)
            print(f"Model loaded from {self.save_dir}")

    def answer_question(self, question):
        question = question.lower()

        if "full form" in question:
            return self.full_form
        elif "address" in question or "location" in question:
            for city in self.location_info:
                if city.lower() in question:
                    return self.location_info[city]
            return "Please specify a center location (Hubli or Belagavi)."
        elif "what is gttc" in question or "more information about gttc" in question:
            return "The Government Tool Room & Training Centre (GTTC) is an educational institution dedicated to offering specialized training in tool and die making, precision manufacturing, and related technical skills. GTTC provides various diploma and certificate courses that equip students with both practical and theoretical knowledge, focusing on areas such as CNC machining, CAD/CAM design, and advanced manufacturing techniques."

        # For other questions, use BERT model
        inputs = self.tokenizer.encode_plus(question, self.context, return_tensors='pt', truncation=True,
                                            max_length=512)
        outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1

        answer_tokens = inputs['input_ids'][0][start_index:end_index]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return answer.strip() if answer else "Sorry, I don't have the answer."

    def calculate_accuracy(self, correct_answers):
        correct_count = 0
        total_questions = len(correct_answers)

        for question, expected_answer in correct_answers.items():
            answer = self.answer_question(question)
            # Normalize answers for comparison
            if expected_answer.lower() in answer.lower():
                correct_count += 1

        accuracy = (correct_count / total_questions) * 100
        return accuracy


# Initialize the QA system
qa_system = GTTCQA()

# Save the model (e.g., after training or modification)
qa_system.save_model()


# Interactive Question-Answering
def query_system():
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = qa_system.answer_question(question)
        print(f"Answer: {answer}")


# Run the interactive query system
if __name__ == "__main__":
    query_system()
