# Variables
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
SRCDIR = src
INCDIR = include
OBJDIR = obj

# Fichiers source
SOURCES = $(SRCDIR)/main.c $(SRCDIR)/neuron.c $(SRCDIR)/network.c $(SRCDIR)/training.c
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TARGET = xor_neural_network

# Règle principale
all: $(OBJDIR) $(TARGET)

# Créer le dossier obj s'il n'existe pas
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Lier l'exécutable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) -lm

# Compiler les fichiers .c en .o
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Nettoyer
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Règles qui ne créent pas de fichiers
.PHONY: all clean
