import re

Token_types = ["Keyword", "Identifier", "Integer", "String", "EndOfToken", "Punctuation", "Operator"]

class Token:
    def __init__(self, type, value):
        if type not in Token_types:
            raise ValueError(f"Invalid token type: {type}")
        self.type = type
        self.value = value

        def get_type(self):
            return self.type

        def get_value(self):
            return self.value
        
def Lexer(input_programme):
    Tokens = []
    keywords = {
        'Keyword': r'(let|in|fn|where|aug|or|not|gr|ge|ls|le|eq|ne|true|false|nil|dummy|within|and|rec)\b',
        'Identifier': r'[a-zA-Z][a-zA-Z0-9_]*',
        'Integer': r'[0-9]+',
        'Operator': r'[+\-*<>&.@/:=~|$\#!%^_\[\]{}"\'?]+',
        'String': r'\'(?:\\\'|[^\'])*\'',
        'Punctuation': r'[();,]',
        'Comment': r'//.*',
        'Spaces': r'[ \t\n]+'
    }

    while input_programme:
        isToken = False
        for token_type, pattern in keywords.items():
            match = re.match(pattern, input_programme)
            if match:
                if token_type == 'Spaces':
                    input_programme = input_programme[match.end():]
                    isToken = True
                    break
                elif token_type == 'Comment':
                    input_programme = input_programme[match.end():]
                    isToken = True
                    break
                else:
                    if token_type not in Token_types:
                        raise ValueError(f"Invalid token type: {token_type}")
                    print(f"Token: {token_type}, Value: {match.group(0)}")
                    Tokens.append(Token(token_type, match.group(0)))
                    input_programme = input_programme[match.end():]
                    isToken = True
                    break
        
        if not isToken:
            print(f"Unexpected character: {input_programme}")
            raise ValueError(f"Unexpected character: {input_programme[0]}")
    
    return Tokens
                

if __name__ == "__main__":
    # Test input string
    test_input = "let x = 5 in x + 3"
    print("Testing lexer with input:", test_input)
    print("\nTokenizing...")
    
    tokens = Lexer(test_input)
    
    print("\nTokens found:")
    for token in tokens:
        print(f"Type: {token.type}, Value: {token.value}")