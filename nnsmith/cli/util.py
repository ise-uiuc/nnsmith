from termcolor import colored


def succ_print(*args):
    return print(*[colored(x, "green") for x in args])


def fail_print(*args):
    return print(*[colored(x, "red") for x in args])


def note_print(*args):
    return print(*[colored(x, "yellow") for x in args])
