from Plaintext_Encrypt import Encrypt
from Plaintext_Decrypt import Decrypt


def main():
    while True:
        # Gives the user three options to encrypt, decrypt, or exit
        x = int(input("1) Encrypt a message\n"
                      "2) Decrypt a message\n"
                      "3) Exit\n"
                      "Response:"))
        if x == 1:
            # User enters text and the alphabet key
            msg = str(input("\nEnter Plaintext Message\n"
                            "Response:"))
            k = str(input("Enter Modified Alphabet Key\n"
                          "Response:"))
            # Displays the encrypted message
            E = Encrypt().encryption(msg, k)
            print("Encrypted Message:", E, "\n")

        elif x == 2:
            # User enters text and the alphabet key used to create the crypto message
            msg = str(input("\nEnter Crypto Message\n"
                            "Response:"))
            k = str(input("Enter Modified Alphabet Key\n"
                          "Response:"))
            # Displays the decrypted message
            D = Decrypt().decryption(msg, k)
            print("Encrypted Message:", D, "\n")
        else:
            break


main()
