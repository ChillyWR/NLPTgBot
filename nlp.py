import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

models: dict = {}
json_data: dict = {}


def clear_replica(replica):
    # функція, яка очищує фразу від символів поза алфавітом, понижає регістр
    replica = replica.lower()
    alphabet = 'йцукенгшщзхъфывапролджэячсмитьбюё '
    result = ''.join(letter for letter in replica if letter in alphabet)
    return result.strip()


def get_intent(replica: str):
    model, vectorizer, train_data = models['default']
    # використовуємо модель, передбачаємо інтент
    replica = clear_replica(replica)
    intent = model.predict(vectorizer.transform([replica]))[0]
    return check_edit_distance_json(replica, intent)


def check_edit_distance_json(replica: str, intent):
    # робимо перевірку використовуючи відстань левенштейна; порівнюємо відстань між прикладами та повідомленням
    for example in json_data['intents'][intent]['examples']:
        example = clear_replica(example)
        distance = nltk.edit_distance(replica, example)
        # змінюючи параметр 0.4 змінимо відношення якість/кількість
        if example and distance / len(example) <= 0.3:
            if intent == '':
                pass
            return intent
    return None


def get_answer_by_intent(intent):
    try:
        if intent in json_data['intents']:
            return random.choice(json_data['intents'][intent]['responses'])
    except IndexError:
        return None


def get_db_intent(replica: str, name):
    model, vectorizer, train_data = models[name]
    # використовуємо модель, передбачаємо інтент
    replica = clear_replica(replica)
    intent = model.predict(vectorizer.transform([replica]))[0]
    return intent


def generate_answer(replica):
    # основна функція для отримання відповіді
    # перше -- розмовні інтенти
    intent = get_intent(replica)
    answer = get_answer_by_intent(intent)
    if answer:
        return answer

    # якщо попередні інтенти не проходять, викликаємо функцію для визначення хвороби
    specialty = get_db_intent(replica, 'specialty')
    disease = get_db_intent(replica, 'disease')

    if specialty:
        answer = f"Вам нужен {specialty}.\n"
    else:
        return failure()
    if disease:
        answer += f"Возможно это {disease}"

    # якщо ф-я не спрацьовує, викликаємо заглушку
    return answer


def failure():
    return random.choice(json_data['failure_phrases'])


def set_json_data(data):
    global json_data
    json_data = data


def set_model(train_data, name: str = ""):
    global models
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
    model = LinearSVC()
    x, y = train_data
    x_vectorazed = vectorizer.fit_transform(x)
    model.fit(x_vectorazed, y)
    models[name] = [model, vectorizer, train_data]
    print("Fitted " + name)
