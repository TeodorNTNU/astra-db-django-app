from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .question_answering import retriever, get_documents


#@require_http_methods(["GET", "POST"])
def ask_question(request):
    if request.method == "POST":
        prompt = request.POST.get('prompt')
        instructions = request.POST.get('qa-instructions', '')
        if prompt:
            answer = retriever(prompt, instructions)
            # Extract the result from the answer dictionary
            result = answer.get('result', 'No result found.')
            context = {'question': prompt, 'answer': result}
            return render(request, 'answer.html', context)
    else:
        documents = get_documents()
        return render(request, 'index.html', {'documents': documents})

def index(request):
    return redirect('ask_question')
