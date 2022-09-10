mkdir -p ~/.streamlit/
echo "\
[server]\n\
email = \"vsumesh0105@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
\n\
" > ~/.streamlit/config.toml


