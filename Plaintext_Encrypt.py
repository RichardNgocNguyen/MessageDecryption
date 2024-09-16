class Encrypt():
    def __init__(self):
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def encryption(self, m, k):
        message = m.upper()
        # Creates a mapping from the alphabet to the modified alphabet
        key = k.upper()
        mapping = dict(zip(self.alphabet, key))
        # Swaps each letter to it's mapping to create an encrypted message
        encrypted_msg = ""
        for letter in message:
            if letter.upper() in self.alphabet:
                encrypted_msg += mapping[letter]
            else:
                encrypted_msg += letter
        return encrypted_msg