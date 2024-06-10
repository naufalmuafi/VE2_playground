# VE2_playground

This repository contains various projects and examples related to reinforcement learning and machine learning, set up with a Webots simulation environment.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Projects](#projects)
  - [1_basic-usage](#1_basic-usage)
  - [2_atari-games](#2_atari-games)
  - [3_tensorboard-integration](#3_tensorboard-integration)
  - [4_custom-env](#4_custom-env)
  - [webots](#webots)
- [License](#license)

## Project Structure

├── .idea
├── 1_basic-usage
├── 2_atari-games
├── 3_tensorboard-integration
├── 4_custom-env
├── pg_venv
├── webots
└── .gitignore

- **.idea**: Project-specific settings for PyCharm.
- **1_basic-usage**: Basic usage examples.
- **2_atari-games**: Examples related to Atari games.
- **3_tensorboard-integration**: TensorBoard integration examples.
- **4_custom-env**: Custom environment setup.
- **pg_venv**: Local virtual environment (ignored in `.gitignore`).
- **webots**: Webots simulation environment setup.
- **.gitignore**: Git ignore file.

## Installation

To get started with this repository, follow these steps:

1. **Clone the repository**:

   ```sh
   git clone https://github.com/yourusername/VE2_playground.git
   cd VE2_playground
   ```

2. **Create and activate a virtual environment**:

   - On Windows:
     ```sh
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To run the examples in this repository, activate the virtual environment and execute the desired scripts. For example:

```sh
python 1_basic-usage/example_script.py
```

## Projects

### 1_basic-usage

Contains basic examples demonstrating the usage of various machine learning techniques.

### 2_atari-games

Examples and projects related to training reinforcement learning agents on Atari games.

### 3_tensorboard-integration

Examples showing how to integrate TensorBoard for tracking and visualizing training metrics.

### 4_custom-env

Custom environment setups for specific projects or experiments.

### webots

Setup and examples for using the Webots robot simulator with reinforcement learning projects.

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
   - Make sure the interpreter is installed correctly.

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
