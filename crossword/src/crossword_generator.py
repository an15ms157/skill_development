# -*- coding: utf-8 -*-
import random
import json
import os
from kivy.metrics import dp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.textfield import MDTextField

class CrosswordGenerator:
    def __init__(self):
        self.bengali_words = []
        self.clues = {}
        self.grid_size = 10
        self.grid = None
        self.placed_words = []
        self.difficulty = "medium"  # Easy, Medium, Hard
        
    def load_bengali_words(self):
        """Load Bengali words and their clues from a predefined list
        In a real app, this would load from a file or database."""
        # Example Bengali words with meanings as clues
        self.bengali_words = [
            {"word": "আম", "clue": "একটি মিষ্টি ফল"},
            {"word": "জল", "clue": "পানীয় পদার্থ"},
            {"word": "বই", "clue": "পড়ার জন্য"},
            {"word": "ফুল", "clue": "সুন্দর গন্ধযুক্ত উদ্ভিদের অংশ"},
            {"word": "চাঁদ", "clue": "রাতের আকাশের আলো"},
            {"word": "সূর্য", "clue": "দিনের আলোর উৎস"},
            {"word": "বাংলা", "clue": "ভাষার নাম"},
            {"word": "বাড়ি", "clue": "যেখানে আমরা থাকি"},
            {"word": "স্কুল", "clue": "শিক্ষা প্রতিষ্ঠান"},
            {"word": "গাছ", "clue": "অক্সিজেন দেয়"},
            {"word": "মাছ", "clue": "জলজ প্রাণী"},
            {"word": "পাখি", "clue": "উড়তে পারে"}
        ]
        
        # Save words by difficulty (could be expanded)
        if self.difficulty == "easy":
            self.bengali_words = [w for w in self.bengali_words if len(w["word"]) <= 3]
        elif self.difficulty == "hard":
            self.bengali_words = [w for w in self.bengali_words if len(w["word"]) >= 4]
    
    def create_empty_grid(self):
        """Create an empty grid of specified size"""
        self.grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
    def can_place_word(self, word, row, col, horizontal):
        """Check if a word can be placed at given position and direction"""
        if horizontal:
            # Out of bounds check
            if col + len(word) > self.grid_size:
                return False
                
            # Check if placement conflicts with existing words
            for i, char in enumerate(word):
                if self.grid[row][col+i] != '.' and self.grid[row][col+i] != char:
                    return False
            
            # Check surrounding cells to ensure we're not adjacent to other words
            # (simplified check - could be made more thorough)
            if col > 0 and self.grid[row][col-1] != '.':
                return False
            if col + len(word) < self.grid_size and self.grid[row][col+len(word)] != '.':
                return False
                
            return True
            
        else:  # vertical
            # Out of bounds check
            if row + len(word) > self.grid_size:
                return False
                
            # Check if placement conflicts with existing words
            for i, char in enumerate(word):
                if self.grid[row+i][col] != '.' and self.grid[row+i][col] != char:
                    return False
                    
            # Check surrounding cells (simplified)
            if row > 0 and self.grid[row-1][col] != '.':
                return False
            if row + len(word) < self.grid_size and self.grid[row+len(word)][col] != '.':
                return False
                
            return True
    
    def place_word(self, word, row, col, horizontal):
        """Place a word on the grid"""
        if horizontal:
            for i, char in enumerate(word):
                self.grid[row][col+i] = char
        else:
            for i, char in enumerate(word):
                self.grid[row+i][col] = char
    
    def generate_crossword(self):
        """Generate a crossword puzzle using the available words"""
        self.create_empty_grid()
        self.placed_words = []
        words = random.sample(self.bengali_words, min(10, len(self.bengali_words)))
        
        # Place first word in the middle, horizontally
        first_word = words[0]["word"]
        mid_row = self.grid_size // 2
        mid_col = (self.grid_size - len(first_word)) // 2
        self.place_word(first_word, mid_row, mid_col, True)
        self.placed_words.append({
            "word": first_word,
            "clue": words[0]["clue"],
            "row": mid_row,
            "col": mid_col,
            "horizontal": True
        })
        
        # Try to place remaining words
        for word_data in words[1:]:
            word = word_data["word"]
            placed = False
            
            # Try to intersect with existing words
            for existing in self.placed_words:
                ex_word = existing["word"]
                ex_row = existing["row"]
                ex_col = existing["col"]
                ex_horizontal = existing["horizontal"]
                
                # Find potential intersection points
                for i, char in enumerate(word):
                    for j, ex_char in enumerate(ex_word):
                        if char == ex_char:
                            # Try horizontal placement
                            if not ex_horizontal:
                                new_row = ex_row + j
                                new_col = ex_col - i
                                if new_col >= 0 and self.can_place_word(word, new_row, new_col, True):
                                    self.place_word(word, new_row, new_col, True)
                                    self.placed_words.append({
                                        "word": word,
                                        "clue": word_data["clue"],
                                        "row": new_row,
                                        "col": new_col,
                                        "horizontal": True
                                    })
                                    placed = True
                                    break
                            
                            # Try vertical placement
                            if ex_horizontal:
                                new_row = ex_row - i
                                new_col = ex_col + j
                                if new_row >= 0 and self.can_place_word(word, new_row, new_col, False):
                                    self.place_word(word, new_row, new_col, False)
                                    self.placed_words.append({
                                        "word": word,
                                        "clue": word_data["clue"],
                                        "row": new_row,
                                        "col": new_col,
                                        "horizontal": False
                                    })
                                    placed = True
                                    break
                        
                    if placed:
                        break
                
                if placed:
                    break
            
            # If we couldn't place this word with intersections, try random placement
            if not placed:
                for _ in range(50):  # Try 50 random positions
                    horizontal = random.choice([True, False])
                    if horizontal:
                        row = random.randint(0, self.grid_size - 1)
                        col = random.randint(0, self.grid_size - len(word))
                    else:
                        row = random.randint(0, self.grid_size - len(word))
                        col = random.randint(0, self.grid_size - 1)
                        
                    if self.can_place_word(word, row, col, horizontal):
                        self.place_word(word, row, col, horizontal)
                        self.placed_words.append({
                            "word": word,
                            "clue": word_data["clue"],
                            "row": row,
                            "col": col,
                            "horizontal": horizontal
                        })
                        placed = True
                        break
        
        # Organize clues by direction
        self.across_clues = []
        self.down_clues = []
        for i, placed in enumerate(self.placed_words):
            clue_num = i + 1
            if placed["horizontal"]:
                self.across_clues.append({
                    "number": clue_num,
                    "clue": placed["clue"],
                    "answer": placed["word"],
                    "row": placed["row"],
                    "col": placed["col"]
                })
            else:
                self.down_clues.append({
                    "number": clue_num,
                    "clue": placed["clue"],
                    "answer": placed["word"],
                    "row": placed["row"],
                    "col": placed["col"]
                })
        
        # Sort clues by number
        self.across_clues.sort(key=lambda x: x["number"])
        self.down_clues.sort(key=lambda x: x["number"])
        
        return self.grid, self.across_clues, self.down_clues
    
    def create_crossword_widgets(self, grid_layout):
        """Create the UI widgets for the crossword grid"""
        grid_layout.clear_widgets()
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == '.':
                    # Empty cell
                    btn = MDFlatButton(
                        text="",
                        size_hint=(None, None),
                        size=(dp(30), dp(30)),
                        md_bg_color=[0, 0, 0, 1]
                    )
                    grid_layout.add_widget(btn)
                else:
                    # Cell with a letter that needs to be filled
                    txt = MDTextField(
                        hint_text="",
                        size_hint=(None, None),
                        size=(dp(30), dp(30)),
                        multiline=False,
                        max_text_length=1,
                        font_name="BengaliFont"
                    )
                    # Store position in the widget for reference
                    txt.pos_in_grid = (row, col)
                    
                    # Add cell number if needed
                    for clue in self.across_clues + self.down_clues:
                        if clue["row"] == row and clue["col"] == col:
                            txt.helper_text = str(clue["number"])
                            txt.helper_text_mode = "persistent"
                            break
                    
                    grid_layout.add_widget(txt)
        
        return grid_layout