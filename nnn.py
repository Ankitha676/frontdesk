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
        Karnataka is one of India's leading technology hubs, with Bengaluru being known as the Silicon Valley of India.
        
        The state has a strong ecosystem in Artificial Intelligence (AI), Machine Learning (ML), Internet of Things (IoT), Robotics, and Generative AI (GenAI). Many global tech companies, startups, and research institutions are based here, making it a major innovation center.
        
        Bengaluru hosts top organizations such as Infosys, Wipro, TCS, and numerous AI startups working on cutting-edge technologies like Agentic AI, autonomous systems, and intelligent automation. The city is also home to premier institutes like IISc Bangalore and IIIT Bangalore, which offer advanced research and training in AI, robotics, and data science.
        
        Popular learning areas in Karnataka include:
        - Artificial Intelligence & Machine Learning
        - Data Science & Analytics
        - Internet of Things (IoT)
        - Robotics & Automation
        - Generative AI (LLMs, RAG, Agentic AI systems)
        - Cloud Computing (AWS, GCP, Azure)
        
        Many institutes and platforms in Karnataka offer training programs, internships, and certifications in these domains. Courses typically include Python programming, deep learning, computer vision, NLP, and AI system design.
        
        The state government also supports tech innovation through initiatives like startup incubators, skill development programs, and smart city projects, encouraging students and professionals to build careers in emerging technologies.
        
        Bengaluru, Mysuru, and Hubballi-Dharwad are growing tech education centers with increasing opportunities in AI and software development.
        
        For anyone interested in technology careers, Karnataka offers strong opportunities in learning, internships, and industry exposure across AI, ML, robotics, and modern software development.
        """
        
        self.full_form = "Karnataka Technology & AI Ecosystem"
        
        self.location_info = {
            "Bengaluru": "Major tech hub with startups, MNCs, and AI research centers",
            "Mysuru": "Growing IT and education hub with training institutes",
            "Hubballi-Dharwad": "Emerging tech and startup ecosystem in North Karnataka"
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
