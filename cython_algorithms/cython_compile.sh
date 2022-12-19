# Compile Cython files (.pyx) into a complete (local) Python extension.
# This is required to use Cython version of crossover and mutation operators for testing.
# If compilation fails, try to run the same command (<python_name_in_your_machine> setup.py build_ext --inplace)
# Solution for multi-OS is taken from: https://stackoverflow.com/questions/394230/how-to-detect-the-os-from-a-bash-script

# Detect Operating System
OS=""
case "$OSTYPE" in
  linux*)
    echo "Using a Linux operating system"
    OS="linux"
    ;;
  msys*)
    echo "Using a Windows operating system"
    OS="windows"
    ;;
  cygwin*)
    echo "Using CygWin for Windows"
    OS="windows"
    ;;
  darwin*)
    echo "Using a MAC operating system"
    OS="mac"
    ;;
  *)
    echo "Using another OS: " $OS
    OS="unknown"
    ;;
esac

if [[ "$1" == "build" ]]; then
  # Build case
  case "$OS" in
    "linux")
      python3 setup.py build_ext --inplace
      ;;
    "windows")
      python setup.py build_ext --inplace
      ;;
    "mac")
      python3 setup.py build_ext --inplace
      ;;
    *)
      python setup.py build_ext --inplace
      ;;
  esac
elif [[ "$1" == "clean" ]]; then
  # Cleaning case
  rm cython_algos.c
  rm cython_algos*.pyd
  rm cython_algos.html
  rm cython_algos*.so
  rm -r build
else
  echo "Unknown command: \"$1\""
fi

read -r -p "Press Enter to finish"