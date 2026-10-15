import re

class ClinicalSupportChatbot:
    """
    Rule-based & pattern-matching conversational chatbot for healthcare and symptom inquiries.
    """
    def __init__(self):
        self.intent_patterns = {
            r"\b(lung cancer|symptom|cough|breath|chest pain)\b": 
                "Common symptoms of lung pathology include persistent cough, shortness of breath, chest pain, and unexplained fatigue. Please consult a qualified pulmonologist for diagnostic imaging.",
            r"\b(heart|cardiac|palpitation|hypertension|blood pressure)\b": 
                "Cardiovascular health indicators include blood pressure, resting heart rate, and cholesterol levels. If experiencing sudden chest pressure or radiating pain, seek immediate emergency medical care.",
            r"\b(appointment|doctor|consultation|schedule)\b": 
                "You can schedule an appointment through your hospital portal or contact our outpatient support desk.",
            r"\b(hello|hi|hey|greetings)\b": 
                "Hello! I am the PathoVision clinical assistant. How may I help you with health information or report interpretation today?"
        }
        self.fallback = "I understand you have a medical query. For specific clinical evaluations, please consult your physician or specialist."

    def get_response(self, user_query):
        query = user_query.lower()
        for pattern, response in self.intent_patterns.items():
            if re.search(pattern, query):
                return response
        return self.fallback
