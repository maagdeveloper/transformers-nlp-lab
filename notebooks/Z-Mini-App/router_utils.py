import joblib

class TfidfRouter:
    def __init__(self, model_path):
        self.router = joblib.load(model_path)

    def predict_route(self, text):
        pred = self.router.predict([text])[0]
        probs = self.router.predict_proba([text])[0]
        scores = {str(k): float(v) for k, v in zip(self.router.classes_, probs)}
        return str(pred), scores

    def route_query(self, text, threshold=0.50):
        label, scores = self.predict_route(text)
        top_score = scores[label]

        if top_score < threshold:
            return {
                "route": "uncertain",
                "predicted_label": label,
                "scores": scores
            }

        if label == "retrieve_generate":
            route = "go_to_rag_pipeline"
        elif label == "direct_qa":
            route = "go_to_qa_pipeline"
        else:
            route = "go_to_chat_pipeline"

        return {
            "route": route,
            "predicted_label": label,
            "scores": scores
        }