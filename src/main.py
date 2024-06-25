from rag.compile import compiled_rag

while 1:
    text_input = input(">> ")
    response = compiled_rag(text_input)
    print(response.answer)
