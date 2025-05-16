# -*- coding: utf-8 -*-
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.clock import Clock
from kivymd.uix.list import OneLineListItem
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
import random  # Added missing import

# Set window size for development
Window.size = (400, 700)

# Register Bengali font
LabelBase.register(name="BengaliFont", 
                  fn_regular="fonts/Noto_Sans_Bengali/NotoSansBengali-Regular.ttf",
                  fn_bold="fonts/Noto_Sans_Bengali/NotoSansBengali-Bold.ttf")

class CrosswordApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.accent_palette = "Amber"
        self.theme_cls.theme_style = "Light"
        
        self.screen_manager = ScreenManager()
        
        # Load screens
        self.home_screen = Builder.load_file("src/screens/home_screen.kv")
        self.game_screen = Builder.load_file("src/screens/game_screen.kv")
        
        # Create screen objects
        home_screen = Screen(name="home")
        home_screen.add_widget(self.home_screen)
        
        game_screen = Screen(name="game")
        game_screen.add_widget(self.game_screen)
        
        # Add screens to manager
        self.screen_manager.add_widget(home_screen)
        self.screen_manager.add_widget(game_screen)
        
        return self.screen_manager
    
    def on_start(self):
        """Initialize the crossword data when app starts"""
        from src.crossword_generator import CrosswordGenerator
        self.crossword_generator = CrosswordGenerator()
        # Pre-load some Bengali words for the crossword
        self.crossword_generator.load_bengali_words()

        # Set up game screen listeners
        self.screen_manager.bind(current=self.on_screen_change)
        
        # Game variables
        self.elapsed_time = 0
        self.timer_active = False
        self.user_solution = {}

    def on_screen_change(self, instance, value):
        """Handle screen change events"""
        if value == "game":
            # Start new game when entering game screen
            self.start_new_game()
        elif value == "home":
            # Stop the timer when returning to home
            self.timer_active = False

    def start_new_game(self):
        """Initialize a new crossword game"""
        # Generate a new crossword
        grid, across_clues, down_clues = self.crossword_generator.generate_crossword()
        
        # Set up the crossword grid in UI
        grid_layout = self.game_screen.ids.crossword_grid
        self.crossword_generator.create_crossword_widgets(grid_layout)
        
        # Add clues to the list
        clue_list = self.game_screen.ids.clue_list
        clue_list.clear_widgets()
        
        # Add horizontal clues
        clue_list.add_widget(OneLineListItem(
            text="বাঁ থেকে ডানে:",
            font_name="BengaliFont",
            font_style="H6"
        ))
        for clue in across_clues:
            clue_list.add_widget(OneLineListItem(
                text="{}. {}".format(clue['number'], clue['clue']),
                font_name="BengaliFont"
            ))
            
        # Add vertical clues
        clue_list.add_widget(OneLineListItem(
            text="উপর থেকে নিচে:",
            font_name="BengaliFont",
            font_style="H6"
        ))
        for clue in down_clues:
            clue_list.add_widget(OneLineListItem(
                text="{}. {}".format(clue['number'], clue['clue']),
                font_name="BengaliFont"
            ))
        
        # Reset timer and start it
        self.elapsed_time = 0
        self.timer_active = True
        Clock.schedule_interval(self.update_timer, 1)
        
        # Initialize user solution tracking
        self.user_solution = {}
    
    def update_timer(self, dt):
        """Update the timer display"""
        if self.timer_active:
            self.elapsed_time += 1
            minutes = self.elapsed_time // 60
            seconds = self.elapsed_time % 60
            self.game_screen.ids.timer_label.text = "{}:{:02d}".format(minutes, seconds)
    
    def toggle_theme(self):
        """Toggle between light and dark theme"""
        if self.theme_cls.theme_style == "Light":
            self.theme_cls.theme_style = "Dark"
        else:
            self.theme_cls.theme_style = "Light"
    
    def provide_hint(self):
        """Provide a hint by revealing a random cell"""
        # Get all the cells in the grid
        grid_layout = self.game_screen.ids.crossword_grid
        cells = [widget for widget in grid_layout.children if hasattr(widget, 'text')]
        
        # Filter for empty cells that should have letters
        empty_cells = [cell for cell in cells if not cell.text]
        
        if empty_cells:
            # Choose a random cell
            cell = random.choice(empty_cells)
            
            # Find the answer for this cell from the crossword generator
            row, col = cell.pos_in_grid  # Assuming we store position in the cell
            correct_letter = self.crossword_generator.grid[row][col]
            
            # Fill in the correct letter
            cell.text = correct_letter
            
            # Show hint dialog
            self.show_dialog(
                "হিন্ট দেওয়া হয়েছে!", 
                "আপনাকে একটি অক্ষর দেওয়া হয়েছে। অবশিষ্ট পাজল পূরণ করুন।",
                "ঠিক আছে"
            )
        else:
            self.show_dialog(
                "কোন হিন্ট নেই", 
                "সমস্ত ঘর ইতিমধ্যে পূরণ করা হয়েছে!",
                "ঠিক আছে"
            )
    
    def check_solution(self):
        """Check if the user's solution is correct"""
        # Get all cells in the grid
        grid_layout = self.game_screen.ids.crossword_grid
        cell_widgets = [w for w in grid_layout.children if hasattr(w, 'text')]
        
        # Check if all cells are filled
        empty_cells = [cell for cell in cell_widgets if not cell.text]
        if empty_cells:
            self.show_dialog(
                "অসম্পূর্ণ সমাধান", 
                "অনুগ্রহ করে সমস্ত ঘর পূরণ করুন।",
                "ঠিক আছে"
            )
            return
        
        # Check if all cells are correct
        all_correct = True
        wrong_cells = []
        
        for row in range(self.crossword_generator.grid_size):
            for col in range(self.crossword_generator.grid_size):
                if self.crossword_generator.grid[row][col] != '.':
                    # Find the corresponding cell widget
                    cell = self.find_cell(cell_widgets, row, col)
                    if cell and cell.text != self.crossword_generator.grid[row][col]:
                        all_correct = False
                        wrong_cells.append(cell)
        
        # Show appropriate dialog
        if all_correct:
            self.timer_active = False
            minutes = self.elapsed_time // 60
            seconds = self.elapsed_time % 60
            self.show_dialog(
                "অভিনন্দন!", 
                "আপনি ক্রসওয়ার্ড সঠিকভাবে সমাধান করেছেন! আপনার সময়: {}:{:02d}".format(minutes, seconds),
                "ঠিক আছে"
            )
        else:
            for cell in wrong_cells:
                cell.error = True
                cell.helper_text = "ভুল"
            
            self.show_dialog(
                "ভুল উত্তর", 
                "আপনার {}টি ভুল আছে। আবার চেষ্টা করুন।".format(len(wrong_cells)),
                "ঠিক আছে"
            )
    
    def find_cell(self, cell_widgets, row, col):
        """Find a specific cell widget by its grid position"""
        for cell in cell_widgets:
            if hasattr(cell, 'pos_in_grid') and cell.pos_in_grid == (row, col):
                return cell
        return None
    
    def show_dialog(self, title, text, button_text):
        """Show a dialog with the given title and text"""
        dialog = MDDialog(
            title=title,
            text=text,
            buttons=[
                MDFlatButton(
                    text=button_text,
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()

if __name__ == "__main__":
    CrosswordApp().run()