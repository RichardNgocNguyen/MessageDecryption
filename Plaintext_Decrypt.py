class Decrypt():
    def __init__(self):
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def decryption(self, m, k):
        message = m.upper()
        # Creates a mapping from the modified alphabet to the alphabet
        key = k.upper()
        mapping = dict(zip(key, self.alphabet))
        # Swaps each letter to it's mapping to decrypt the message
        decrypted_msg = ""
        for letter in message:
            if letter.upper() in self.alphabet:
                decrypted_msg += mapping[letter]
            else:
                decrypted_msg += letter
        return decrypted_msg