# Bits & Pieces

A minimalistic Jekyll blog with Bauhaus-inspired design for documenting hardware and software projects.

## Features

- **Bauhaus-inspired design** - Clean, geometric, and functional
- **Responsive layout** - Works perfectly on all devices
- **Fast and lightweight** - Optimized for performance
- **SEO optimized** - Built-in meta tags and structured data
- **RSS feed** - Automatic feed generation
- **Dark/light mode ready** - CSS custom properties for easy theming

## Design Philosophy

This site embodies the Bauhaus principles of "form follows function" with a playful twist:

- **Geometric clarity** - Clean lines and structured layouts
- **Bold typography** - Clear hierarchy and readability
- **Functional minimalism** - Every element serves a purpose
- **Playful color palette** - Vibrant oranges, deep blues, and warm accents

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

## Customization

### Colors

The color scheme is defined in CSS custom properties in `assets/css/main.css`:

```css
:root {
  --color-primary: #ff6b35;     /* Vibrant orange */
  --color-secondary: #004e89;   /* Deep blue */
  --color-accent: #f7931e;      /* Warm yellow-orange */
  --color-neutral: #2c3e50;     /* Dark gray-blue */
  --color-light: #ecf0f1;       /* Light gray */
}
```

### Typography

The site uses Inter as the primary font with JetBrains Mono for code:

```css
:root {
  --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', monospace;
}
```

### Adding Posts

Create new posts in the `_posts/` directory with the following front matter:

```markdown
---
layout: post
title: "Your Post Title"
date: 2024-01-15
excerpt: "A brief description of your post"
tags: [tag1, tag2, tag3]
---

Your post content here...
```

### Adding Pages

Create new pages in the root directory with the following front matter:

```markdown
---
layout: page
title: Page Title
permalink: /page-url/
---

Your page content here...
```

## Deployment

### GitHub Pages

1. Push your code to a GitHub repository
2. Go to Settings > Pages
3. Select "Deploy from a branch"
4. Choose the `main` branch and `/ (root)` folder
5. Your site will be available at `https://username.github.io/repository-name`

### Netlify

1. Connect your GitHub repository to Netlify
2. Set build command: `bundle exec jekyll build`
3. Set publish directory: `_site`
4. Deploy!

### Vercel

1. Connect your GitHub repository to Vercel
2. Vercel will automatically detect Jekyll
3. Deploy!

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Bauhaus design principles
- Built with Jekyll
- Typography by Inter and JetBrains Mono
- Icons and graphics created with geometric precision

---

*Built with ❤️ and a lot of coffee.* 