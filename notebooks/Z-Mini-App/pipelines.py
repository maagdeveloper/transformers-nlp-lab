from router_utils import TfidfRouter
from qa_utils import qa_pipeline
from chat_utils import chat_pipeline
from rag_utils import rag_pipeline

router = TfidfRouter("../TF-IDF router/router_tfidf.pkl")

def handle_query(text):
    result = router.route_query(text)

    if result["route"] == "go_to_qa_pipeline":
        return {"router": result, "output": qa_pipeline(text)}

    if result["route"] == "go_to_chat_pipeline":
        return {"router": result, "output": chat_pipeline(text)}

    if result["route"] == "go_to_rag_pipeline":
        return {"router": result, "output": rag_pipeline(text)}

    return {
        "router": result,
        "output": {
            "type": "uncertain",
            "message": "No pipeline selected."
        }
    }