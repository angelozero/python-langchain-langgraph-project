from rich import print
from factory import get_chat_model

from langchain_core.messages import HumanMessage, SystemMessage


def main():
    llm = get_chat_model()

    # Prompt para enviarmos para IA para que ela tenha um comportamento especifico
    system_message = SystemMessage(
        """ 
        Você é um assistente astronomo. 
        Seu objetivo é me ajudar a entender o sistema solar. 
        Use um tom de facil entendimento e forneça a resposta em formato de texto_curto.
        """
    )
    
    # Mensagem de pergunta humana
    human_message = HumanMessage("Qual é o planeta mais próximo ao sol?")
    
    messages = [system_message, human_message]

    response = llm.invoke(messages)
    
    print(response)


if __name__ == "__main__":
    main()
