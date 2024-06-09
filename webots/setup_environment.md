# Setting Up PyCharm in Windows for a Webots Project

1. **Navigate to the Project Directory**:

   - Open Command Prompt and navigate to your Webots project directory.
     ```sh
     cd path\to\your\webots\project
     ```

2. **Create a Virtual Environment**:

   - Run the following command to create a virtual environment:
     ```sh
     python -m venv venv
     ```

3. **Launch PyCharm**:

   - Open PyCharm.

4. **Configure Project Structure**:

   - Go to `File > Settings > Project: <Your Project> > Project Structure`.
   - Click `Add Content Root` and select `WEBOTS_HOME/lib/controller/python`.

5. **Edit Configuration**:

   - Click `Edit Configurations` on the top right (next to the run/debug configurations dropdown).
   - Click the `+` icon and select `Python`.
   - In the `Name` field, give your configuration a name.
   - In the `Script` field, select your Python controller script.
   - In the `Environment variables` section, click `...` and then `+` to add a new environment variable.
     - Set the `Name` to `Path`.
     - Set the `Value` to `F:\Program Files\Webots\lib\controller\;F:\Program Files\Webots\msys64\mingw64\bin\;F:\Program Files\Webots\msys64\mingw64\bin\cpp`.

6. **Develop Your Program**:

   - Start editing and developing your program within PyCharm.

7. **Install `pip-tools`**:

   - Open the terminal in PyCharm and run:
     ```sh
     pip install pip-tools
     ```

8. **Create `requirements.in` File**:

   - Create a file named `requirements.in` in your project directory.
   - List the required libraries in this file and save it.

9. **Compile and Sync Requirements**:

   - Run the following commands in the terminal:
     ```sh
     pip-compile
     pip-sync
     ```

10. **Run from IDE**:
    - In Webots, select the `<extern>` option to run the controller script from the IDE.

# Setting Up VS Code in Windows for a Webots Project

1. **Navigate to the Project Directory**:

   - Open Command Prompt and navigate to your Webots project directory.
     ```sh
     cd path\to\your\webots\project
     ```

2. **Create a Virtual Environment**:

   - Run the following command to create a virtual environment:
     ```sh
     python -m venv venv
     ```

3. **Open VS Code**:

   - Open Visual Studio Code.

4. **Open the Project Folder**:

   - In VS Code, go to `File > Open Folder` and select your Webots project directory.

5. **Configure the Python Interpreter**:

   - Press `Ctrl+Shift+P` to open the Command Palette.
   - Type `Python: Select Interpreter` and select the interpreter from your virtual environment. It should be something like `./venv/Scripts/python`.

6. **Install Required Extensions**:

   - Ensure you have the Python extension installed in VS Code. If not, install it from the Extensions view (`Ctrl+Shift+X`).

7. **Add `WEBOTS_HOME/lib/controller/python` to the Workspace**:

   - In VS Code, go to `File > Add Folder to Workspace...`.
   - Navigate to `WEBOTS_HOME/lib/controller/python` and add it to your workspace. This will allow VS Code to recognize the additional Python libraries provided by Webots.

8. **Save the Workspace Configuration**:

   - Go to `File > Save Workspace As...` and save the workspace configuration with a name like `webots-project.code-workspace`.

9. **Configure Environment Variables**:

   - Create a `.env` file in the root of your project directory.
   - Add the following line to the `.env` file:
     ```sh
     DYLD_LIBRARY_PATH=WEBOTS_HOME/lib/controller
     ```

10. **Set Up Launch Configuration**:

    - In VS Code, go to `Run > Add Configuration`.
    - Choose `Python` and then `Python File`.
    - Modify the generated `launch.json` to include your script and environment variables:
      ```json
      {
        "version": "0.2.0",
        "configurations": [
          {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "env": {
              "PYTHONPATH": "${workspaceFolder}/WEBOTS_HOME/lib/controller/python"
            }
          }
        ]
      }
      ```

11. **Develop Your Program**:

    - Start editing and developing your program within VS Code.

12. **Install `pip-tools`**:

    - Open the integrated terminal in VS Code (`Ctrl+``).
    - Run the following command to install `pip-tools`:
      ```sh
      pip install pip-tools
      ```

13. **Create `requirements.in` File**:

    - Create a file named `requirements.in` in your project directory.
    - List the required libraries in this file and save it.

14. **Compile and Sync Requirements**:

    - Run the following commands in the terminal:
      ```sh
      pip-compile
      pip-sync
      ```

15. **Run from IDE**:
    - In Webots, select the `<extern>` option to run the controller script from the IDE.
