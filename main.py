import customtkinter as ctk
from generator import generate

ctk.set_default_color_theme("dark-blue") # blue, green, dark-blue
ctk.set_appearance_mode("system") # dark, light, system

class Voxel3(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry("550x250")
        self.title("Voxel3")

        # 1. Prompt reception
        self.lbl_prompt = ctk.CTkLabel(self, text="Type in your 3D model idea down here:")
        self.lbl_prompt.pack(pady=20)

        self.prompt_entry = ctk.CTkEntry(self, placeholder_text="A red birthday cake...")
        self.prompt_entry.pack(pady=15)

        # 2. Confirmation and execution
        self.btn_confirmation = ctk.CTkButton(self, text="Generate", command=self.generate)
        self.btn_confirmation.pack(pady=10)

        # 3. Warnings 
        self.lbl_warning = ctk.CTkLabel(self, text="", text_color="red")
        self.lbl_warning.pack(pady=10)

    def generate(self):
        prompt = self.prompt_entry.get()

        if not prompt:
            self.lbl_warning.configure(text="ERROR: Please type in a prompt.")
            return
        else:
            print(f"Starting generation of {prompt}.")
            generate(prompt)


if __name__ == "__main__":
    app = Voxel3()
    app.mainloop()