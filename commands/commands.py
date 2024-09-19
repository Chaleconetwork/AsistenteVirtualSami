import webbrowser

commands = ['abre', 'hola', 'ruido', 'sami', 'youtube']
httpCommands = ['youtube']

def open_trigger(command):
    if 'abre' in command:
        command.remove('abre')

    if command[0] in httpCommands and len(command) == 1:
        url = f'https://www.{command[0]}.com'
        webbrowser.open(url)
        print(command[0])