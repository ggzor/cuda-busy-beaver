#!/usr/bin/env python3

import argparse
from itertools import filterfalse
from operator import itemgetter
from collections import defaultdict, namedtuple
import re

token_separator = re.compile(r'(\(|,|\))')
def tokenize(s):
    return list(filterfalse(token_separator.match, filter(None, map(str.strip, token_separator.split(s)))))

Zero = namedtuple('Zero', 'register')
Succesor = namedtuple('Succesor', 'register')
Transfer = namedtuple('Transfer', 'source,destination')
Jump = namedtuple('Jump', 'left,right,destination')
Address = namedtuple('Address', 'name')

kind_mappings = { 
    'Z': Zero,
    'S': Succesor,
    'T': Transfer,
    'J': Jump
}

def parse_instruction(s):
    kind = s[0]

    if kind in { 'Z', 'S' }:
        params = s[1:2]
        addresses = s[2:]
    elif kind == 'T':
        params = s[1:3]
        addresses = s[3:]
    elif kind == 'J':
        params = s[1:4]
        addresses = s[4:]

    def fix_parameter(s):
        try:
            return int(s)
        except:
            return Address(s)

    params = list(map(fix_parameter, params))
    addresses = list(map(Address, addresses))

    return kind_mappings[kind], params, addresses


def resolve_references(instructions):
    references = {}

    for i, addresses in filter(itemgetter(1), enumerate(map(itemgetter(2), instructions), 1)):
        for addr in addresses:
            references[addr] = i

    def resolve_parameter_refs(parameter):
        if isinstance(parameter, int):
            return parameter
        elif isinstance(parameter, Address):
            return references[parameter]

    original_refs = {}
    final_instructions = []

    for i, instruction_data in enumerate(instructions):
        kind, params, _ = instruction_data

        if kind == Jump and isinstance(params[2], Address):
            original_refs[i] = params[2]

        final_instructions.append(kind(*map(resolve_parameter_refs, params)))


    return final_instructions, references, original_refs


def parse_file(s):
    return resolve_references(list(map(parse_instruction, filterfalse(lambda s:s[0].startswith('#'), filter(None, map(tokenize, s.splitlines()))))))

def execute_raw_urm(program_data, R):
    instructions, references, original_refs = program_data

    def print_registers():
        maximum = max([1, *R])

        print(' '.join(f'{R[i]}' for i in range(1, maximum + 1)))

    def pretty_print_instruction(instruction, idx, refs):
        if isinstance(instruction, Zero):
            return f'Z({instruction.register})'
        elif isinstance(instruction, Succesor):
            return f'S({instruction.register})'
        elif isinstance(instruction, Transfer):
            return f'T({instruction.source}, {instruction.destination})'
        elif isinstance(instruction, Jump):
            if idx in refs:
                destination = refs[idx].name
            else:
                destination = instruction.destination

            return f'J({instruction.left}, {instruction.right}, {destination})'

    i = 0
    count = 0

    while 0 <= i < len(instructions):
        count += 1
        instruction = instructions[i]

        print_registers()
        print(f'Next: {pretty_print_instruction(instruction, i, original_refs)}')

        if isinstance(instruction, Zero):
            R[instruction.register] = 0
        elif isinstance(instruction, Succesor):
            R[instruction.register] += 1
        elif isinstance(instruction, Transfer):
            R[instruction.destination] = R[instruction.source]
        elif isinstance(instruction, Jump):
            if R[instruction.left] == R[instruction.right]:
                i = instruction.destination - 1
                continue

        i += 1

    print(f'\nFinal configuration after {count} instructions:')
    print_registers()

    print('\nResult:', R[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='urm', description='e-URM Interpreter')
    parser.add_argument('file', type=argparse.FileType('r'), help='The file to interpret', metavar='FILE')
    parser.add_argument('initial', type=int, nargs='*', help='The initial configuration register values', metavar='R')
    args = parser.parse_args()

    contents = args.file.read()
    program_data = parse_file(contents)
    execute_raw_urm(program_data, defaultdict(lambda: 0, { i : int(v) for i, v in enumerate(args.initial, 1) }))
