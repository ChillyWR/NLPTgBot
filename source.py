import input_parser
import nlp
import bot


def main():
    data = input_parser.load_data()
    nlp.set_json_data(data[0])
    nlp.set_model(input_parser.get_json_td(data[0]), 'default')
    nlp.set_model(input_parser.get_disease_td(), 'disease')
    nlp.set_model(input_parser.get_specialty_td(), 'specialty')
    bot.start()


if __name__ == '__main__':
    main()


