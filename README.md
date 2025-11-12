# Bits & Pieces

## Quick Start

### Prerequisites

- Ruby 2.7 or higher
- RubyGems
- GCC and Make

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aliroberts/bitsnpieces.dev.git
   cd bitsnpieces.dev
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Start the development server**
   ```bash
   bundle exec jekyll serve
   ```

4. **Visit your site**
   Open [http://localhost:4000](http://localhost:4000) in your browser

## Project Structure

```
├── _config.yml          # Jekyll configuration
├── _layouts/            # HTML templates
│   ├── default.html     # Main layout
│   ├── post.html        # Blog post layout
│   └── page.html        # Static page layout
├── _posts/              # Blog posts (Markdown)
├── assets/              # Static assets
│   ├── css/            # Stylesheets
│   └── favicon.svg     # Site favicon
├── about.md            # About page
├── index.html          # Home page
└── README.md           # This file
```