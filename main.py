# --- START OF FILE pokemon_chatbot_final_en_hr_comments.py ---

# Uvoz potrebnih modula
import traceback  # Za ispis detaljnih informacija o iznimkama
from io import BytesIO  # Rukovanje binarnim podacima slike u memoriji
import base64      # Kodiranje slike u Base64 string
import numpy as np  # Numeričke operacije za grafikon
import matplotlib.pyplot as plt
import os
import json
import requests  # Za HTTP zahtjeve prema PokeAPI
import gradio as gr  # Za korisničko sučelje

# Pretpostavka postojanja konfiguracijske datoteke za API ključeve
from ai_config import Clients, verify_api_keys

# Uvoz modula za generiranje grafikona
import matplotlib
matplotlib.use('Agg')  # Postavljanje ne-GUI backend-a za Matplotlib

# --- Konfiguracija ---
# Verifikacija API ključeva pri pokretanju
verify_api_keys()
# Inicijalizacija OpenAI klijenta iz konfiguracije
clients = Clients()
openai = clients['openai']
# Definicija modela koji će se koristiti
GPT_MODEL = 'gpt-4o-mini'

# --- Sistemska Poruka (Upute za LLM) ---
# Definira ulogu i pravila ponašanja za LLM, s naglaskom na obradu alata i fokusu na zadnji upit
system_message = (
    "You are PokéPal AI, a helpful Pokémon expert.\n"
    # Naglasak na kontekstu zadnje poruke
    "**IMPORTANT: Your goal is to answer based *only* on the user's *most recent* message.**\n\n"
    "When a user asks about specific Pokémon or asks to compare Pokémon in their **latest message**:\n"
    # Preciziranje izvora imena
    "1. **Identify ALL Pokémon names mentioned *only* within that single, most recent user message.**\n"
    "2. Use the `get_pokemon_info` tool. Provide the list of names identified in step 1 in the `pokemon_names` parameter.\n"
    # Eksplicitna zabrana dohvaćanja iz povijesti za alat
    "   **CRITICAL: Do NOT include Pokémon names from previous messages in the conversation unless they are explicitly repeated in the *current* user message.**\n"
    "3. Use the information successfully retrieved by the tool for the currently requested Pokémon to formulate your textual response.\n"
    "4. A comparison graph comparing *only the Pokémon successfully retrieved for the current request* will be generated automatically if multiple results are available. A single graph might appear for single Pokémon requests. Comment on the data shown in the graph relevant *only* to the current request.\n"  # Kontekstualizacija grafikona i odgovora
    "5. If the tool returns an error for any Pokémon mentioned in the current request, inform the user politely.\n\n"
    "Focus strictly on fulfilling the user's latest request."  # Završni naglasak
)

# --- Implementacija Alata: Funkcija za Dohvaćanje Pokémon Informacija ---
# Ova funkcija komunicira s vanjskim API-jem (PokeAPI)


def get_pokemon_info(pokemon_names: list):
    """
    Dohvaća podatke za listu Pokémona s PokeAPI-ja.
    Args: pokemon_names (list): Lista imena Pokémona (npr. ["pikachu", "charizard"]).
    Returns: Rječnik gdje su ključevi imena Pokémona, a vrijednosti rječnici s podacima ili rječnici s greškom.
    """
    print(f"--- TOOL CALL: get_pokemon_info for {pokemon_names} ---")
    # Rječnik za agregaciju rezultata dohvaćanja za svaki traženi Pokémon
    results = {}
    if not pokemon_names:
        # Validacija ulaznog popisa imena
        return {"error": "No Pokémon names were provided."}

    # Definiranje standardnog redoslijeda statistika za konzistentnost grafikona
    stat_order = ["hp", "attack", "defense",
                  "special-attack", "special-defense", "speed"]

    # Iteracija kroz listu zadanih imena Pokémona
    for name in pokemon_names:
        # Osnovna validacija pojedinačnog imena
        if not name or not isinstance(name, str):
            results[str(name)] = {"error": "Invalid name provided in list."}
            continue

        # Normalizacija imena (mala slova) za API poziv
        pokemon_name_lower = name.lower()
        # Konstrukcija URL-a za PokeAPI endpoint
        api_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name_lower}"
        print(f"--- Fetching data for: {pokemon_name_lower} ---")

        try:
            # Slanje HTTP GET zahtjeva prema API-ju s definiranim timeout-om
            response = requests.get(api_url, timeout=10)
            # Provjera HTTP status koda; podiže HTTPError za loše odgovore (4xx ili 5xx)
            response.raise_for_status()
            # Parsiranje JSON odgovora u Python rječnik
            data = response.json()

            # --- Ekstrakcija Statistika (Stats) ---
            # Dohvat liste statsa ili prazne liste
            stats_data = data.get("stats", [])
            extracted_stats = {}
            # Kreiranje privremenog rječnika za lakši pristup statsu po imenu
            temp_stats = {s["stat"]["name"]: s["base_stat"]
                          for s in stats_data}
            # Iteriranje kroz definirani redoslijed statsa i popunjavanje rječnika
            for stat_name in stat_order:
                # Default vrijednost 0 ako stat nedostaje
                extracted_stats[stat_name] = temp_stats.get(stat_name, 0)

            # --- Ekstrakcija Ostalih Informacija ---
            info = {
                "name": data.get("name"),
                "id": data.get("id"),
                # Ekstrakcija imena tipova iz liste objekata
                "types": [t["type"]["name"] for t in data.get("types", [])],
                # Ekstrakcija imena sposobnosti iz liste objekata
                "abilities": [a["ability"]["name"] for a in data.get("abilities", [])],
                "height": data.get("height"),  # Jedinica: decimetri
                "weight": data.get("weight"),  # Jedinica: hektogrami
                "base_experience": data.get("base_experience"),
                "stats": extracted_stats,  # Dodavanje rječnika s ekstrahiranim statsima
                # URL do defaultnog sprite-a
                "sprite_url": data.get("sprites", {}).get("front_default")
            }
            # Pohrana uspješno dohvaćenih podataka u rezultatski rječnik
            results[name] = info
            print(f"--- TOOL SUCCESS for: {name} ---")

        # --- Obrada Potencijalnih Grešaka ---
        except requests.exceptions.HTTPError as e:
            # Specifična obrada za 404 (Not Found) grešku
            error_msg = f"Pokémon '{name}' not found." if e.response.status_code == 404 else f"HTTP error for '{name}': {e}"
            results[name] = {"error": error_msg}
            print(f"--- TOOL ERROR for {name}: {error_msg} ---")
        except requests.exceptions.RequestException as e:
            # Obrada mrežnih grešaka (npr. DNS, timeout, konekcija)
            error_msg = f"Network error for '{name}': {e}"
            results[name] = {"error": error_msg}
            print(f"--- TOOL ERROR for {name}: {error_msg} ---")
        except Exception as e:
            # Obrada svih ostalih neočekivanih iznimki
            error_msg = f"Unexpected error for '{name}': {e}"
            results[name] = {"error": error_msg}
            print(f"--- TOOL ERROR for {name}: {error_msg} ---")

    # Vraćanje rječnika koji sadrži rezultate (uspješne ili greške) za sve tražene Pokémone
    return results


# --- Opis Alata (JSON Schema) za OpenAI Model ---
# Definira strukturu, parametre i svrhu alata `get_pokemon_info` za LLM.
pokemon_tool_description = {
    "name": "get_pokemon_info",  # Ime mora odgovarati Python funkciji
    "description": "Get detailed information (types, abilities, base stats) for one or more Pokémon. Use this for single Pokémon queries or when the user asks to **compare** multiple Pokémon.",
    "parameters": {
        "type": "object",
        "properties": {
            # Parametar je sada lista stringova umjesto jednog stringa
            "pokemon_names": {
                "type": "array",  # Označava da je vrijednost lista/polje
                "items": {
                    "type": "string"  # Specificira tip elemenata unutar liste
                },
                "description": "A list of the names of the Pokémon to look up (e.g., ['pikachu'] or ['pikachu', 'charizard', 'mewtwo']). Extract all relevant names from the user's query.",
            }
        },
        # Ovaj parametar je obavezan za poziv alata
        "required": ["pokemon_names"],
        "additionalProperties": False  # Sprječava LLM da dodaje nedefinirane parametre
    },
}

# --- Lista Alata dostupnih LLM-u ---
# Sadrži sve alate koje LLM može koristiti; trenutno samo jedan.
tools = [{"type": "function", "function": pokemon_tool_description}]

# --- Funkcija za Generiranje Spider Grafikona ---


def create_comparison_spider_chart(pokemon_data: dict):
    """
    Generira usporedni spider (radar) grafikon za statse jednog ili više Pokémona.
    Args: pokemon_data (dict): Rječnik gdje su ključevi imena Pokémona, a vrijednosti njihovi podaci (moraju sadržavati 'stats').
    Returns: Base64 kodirani string PNG slike, ili None ako podaci nisu dovoljni ili dođe do greške.
    """
    # Validacija ulaznih podataka
    if not pokemon_data or len(pokemon_data) < 1:
        return None

    # Priprema podataka za crtanje
    pokemon_names = list(pokemon_data.keys())
    # Dohvaćanje labela (imena statsa) iz prvog uspješnog rezultata
    first_pokemon_stats = next(iter(pokemon_data.values())).get("stats", {})
    labels = list(first_pokemon_stats.keys())
    num_vars = len(labels)
    # Ako nema statsa, ne možemo crtati
    if num_vars == 0:
        return None

    # Izračun kutova za svaku os grafikona (jednako raspoređeni)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Zatvaranje kruga dodavanjem prvog kuta na kraj

    # Definicija palete boja za različite Pokémone
    colors = ['skyblue', 'salmon', 'lightgreen',
              'gold', 'plum', 'lightcoral', 'orange', 'cyan']

    # Stvaranje Matplotlib figure i polarnih osi
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Iteracija kroz podatke za svakog Pokémona i crtanje
    for i, name in enumerate(pokemon_names):
        stats_dict = pokemon_data[name].get("stats", {})
        # Ekstrakcija vrijednosti statsa prema definiranom redoslijedu
        stats = [stats_dict.get(label, 0) for label in labels]
        stats += stats[:1]  # Zatvaranje niza podataka za crtanje
        # Odabir boje iz palete (ciklički)
        color = colors[i % len(colors)]

        # Crtanje linije statsa i ispunjavanje područja ispod nje
        ax.plot(angles, stats, linewidth=1.5, linestyle='solid', color=color,
                label=name.capitalize())  # label se koristi u legendi
        ax.fill(angles, stats, color, alpha=0.25)  # Prozirna ispuna

    # Postavljanje vizualnih elemenata grafikona
    ax.set_ylim(0, 180)  # Postavljanje maksimuma Y-osi (raspon statsa)
    ax.set_xticks(angles[:-1])  # Postavljanje pozicija za labele osi
    # Postavljanje teksta labela osi (imena statsa), s formatiranjem
    ax.set_xticklabels([label.replace('-', ' ').title() for label in labels])
    # Postavljanje numeričkih oznaka (grida) na osima
    ax.set_yticks(np.arange(0, 181, 30))  # Korak od 30
    ax.set_yticklabels([str(x) for x in np.arange(0, 181, 30)])
    # Dodavanje razmaka između labela osi i grafa
    ax.tick_params(axis='x', pad=10)

    # Postavljanje naslova i legende, s prilagodbom pozicije
    if len(pokemon_names) > 1:
        ax.set_title("Base Stat Comparison", size=14, y=1.12)
        # Pozicioniranje legende izvan područja crtanja za usporedni graf
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    else:
        # Naslov za grafikon s jednim Pokémonom
        ax.set_title(
            f"{pokemon_names[0].capitalize()} Base Stats", size=14, y=1.12)
        # Pozicioniranje legende na 'najbolje' mjesto za jednostruki graf
        ax.legend(loc='best')

    # Spremanje generiranog grafikona u memorijski buffer
    buf = BytesIO()
    image_base64 = None  # Inicijalizacija
    try:
        # Spremanje figure u buffer kao PNG, uz optimizaciju veličine (dpi)
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=90)
        buf.seek(0)  # Premotavanje buffera na početak za čitanje
        # Kodiranje binarnih podataka slike u Base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Error saving figure: {e}")
        # image_base64 ostaje None
    finally:
        # Zatvaranje Matplotlib figure radi oslobađanja memorije
        plt.close(fig)

    # Vraćanje Base64 stringa slike
    return image_base64

# --- Funkcija za Obradu Poziva Alata od Strane LLM-a ---
# Upravlja izvršavanjem odgovarajuće Python funkcije kada LLM zatraži alat.


def handle_tool_calls(message):
    """Obrađuje zahtjeve za poziv alata iz LLM poruke."""
    # Lista za pohranu formatiranih odgovora alata za LLM
    tool_responses = []
    # Mapiranje imena alata (kako ih LLM zove) na stvarne Python funkcije
    available_functions = {"get_pokemon_info": get_pokemon_info}

    # Iteracija kroz sve zahtjeve za alat u LLM poruci
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name  # Ime tražene funkcije
        function_to_call = available_functions.get(
            function_name)  # Dohvat Python funkcije
        # Parsiranje JSON stringa s argumentima koje je LLM pripremio
        function_args = json.loads(tool_call.function.arguments)

        print(
            f"--- LLM requested call to: {function_name} with args: {function_args} ---")

        if not function_to_call:
            # Slučaj kada LLM zatraži nepostojeći alat
            print(f"Error: LLM requested unknown function '{function_name}'")
            function_response_content = json.dumps(
                {"error": f"Unknown tool '{function_name}' requested."})
        else:
            # Pokušaj izvršavanja mapirane Python funkcije
            try:
                # Poziv stvarne funkcije (`get_pokemon_info`) s argumentima iz LLM-a
                function_response = function_to_call(
                    # Prosljeđivanje liste imena iz parsiranih argumenata
                    pokemon_names=function_args.get(
                        "pokemon_names", [])  # Očekuje se lista
                )
                # Pretvaranje rezultata funkcije (rječnika) u JSON string za LLM
                function_response_content = json.dumps(function_response)
            except Exception as e:
                # Hvatanje iznimki tijekom izvršavanja alata
                print(f"Error executing function {function_name}: {e}")
                function_response_content = json.dumps(
                    {"error": f"Error executing tool {function_name}: {str(e)}"})

        # Dodavanje formatiranog odgovora alata u listu za LLM
        tool_responses.append({
            "role": "tool",  # Označava da je ovo odgovor od alata
            "content": function_response_content,  # Sadržaj je JSON string s rezultatima
            "tool_call_id": tool_call.id  # Poveznica na originalni zahtjev LLM-a
        })
    # Vraćanje liste svih odgovora alata
    return tool_responses

# --- Glavna Chat Funkcija ---
# Orkestrira cijeli proces: obrada korisničke poruke, pozivi LLM-u, rukovanje alatima i grafikonom.


def chat(message, history):
    """Glavna funkcija za obradu korisničke poruke i generiranje odgovora."""
    print(f"\nUser: {message}")
    # 1. Konstrukcija liste poruka za LLM: sistemska poruka + povijest + nova poruka
    messages = [{"role": "system", "content": system_message}]
    # Dodavanje poruka iz povijesti razgovora
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        # Osnovna provjera da se izbjegne ponovno slanje Markdown slika LLM-u
        if "![Comparison Stats Graph]" not in ai and "![Stats Graph]" not in ai:
            messages.append({"role": "assistant", "content": ai})

    # Dodavanje najnovije korisničke poruke
    messages.append({"role": "user", "content": message})

    # Inicijalizacija varijabli
    base64_image = None  # Za spremanje Base64 stringa grafikona
    final_response_text = "An error occurred."  # Defaultni odgovor u slučaju greške

    try:
        # 2. Prvi poziv OpenAI API-ju
        # LLM može odlučiti odgovoriti direktno ili zatražiti poziv alata
        print("--- Calling OpenAI (potential tool use) ---")
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            tools=tools,        # Prosljeđivanje definicije alata
            tool_choice="auto"  # LLM odlučuje hoće li koristiti alat
        )

        # Dohvaćanje odgovora LLM-a
        response_message = response.choices[0].message

        # 3. Provjera je li LLM zatražio poziv alata
        if response_message.tool_calls:
            print(
                f"--- LLM requested tool call(s): {response_message.tool_calls} ---")
            # 4. Dodavanje LLM-ovog zahtjeva za alat u listu poruka za kontekst
            messages.append(response_message)

            # 5. Izvršavanje traženih alata pomoću `handle_tool_calls`
            tool_call_responses = handle_tool_calls(response_message)

            # --- Obrada rezultata alata i priprema podataka za grafikon ---
            # Za sve rezultate (uključujući greške)
            all_pokemon_data_from_tool = {}
            successful_pokemon_data_for_graph = {}  # Samo uspješni rezultati sa statsima

            # Iteracija kroz odgovore alata
            for tool_response in tool_call_responses:
                if tool_response["role"] == "tool":
                    try:
                        # Parsiranje JSON stringa iz content polja
                        content_data = json.loads(tool_response["content"])
                        # Agregacija rezultata
                        if isinstance(content_data, dict):
                            all_pokemon_data_from_tool.update(content_data)
                    except json.JSONDecodeError:
                        print("Warning: Could not decode tool response content.")
                    except Exception as e:
                        print(
                            f"Warning: Error processing tool response content: {e}")

            # Filtriranje samo uspješnih rezultata koji sadrže 'stats' ključ
            for name, data in all_pokemon_data_from_tool.items():
                if isinstance(data, dict) and "error" not in data and "stats" in data:
                    # Korištenje normaliziranog imena kao ključa
                    successful_pokemon_data_for_graph[name.lower()] = data

            print(
                f"--- Data prepared for graphing: {list(successful_pokemon_data_for_graph.keys())} ---")

            # --- Generiranje grafikona ako postoji barem jedan uspješan rezultat ---
            num_successful = len(successful_pokemon_data_for_graph)
            graph_label = "Stats Graph"  # Defaultna oznaka za sliku

            # Uvjet promijenjen na >= 1 za prikaz grafa i za pojedinačne Pokémone
            if num_successful >= 1:
                # Postavljanje specifične oznake ovisno o broju rezultata
                if num_successful > 1:
                    graph_label = "Comparison Stats Graph"
                    print(
                        f"--- Generating Comparison Spider Chart for {num_successful} Pokémon ---")
                else:
                    pokemon_name_single = list(
                        successful_pokemon_data_for_graph.keys())[0]
                    graph_label = f"{pokemon_name_single.capitalize()} Stats Graph"
                    print(
                        f"--- Generating Single Spider Chart for {pokemon_name_single} ---")

                # Pokušaj generiranja grafikona
                try:
                    # Poziv funkcije za crtanje
                    base64_image = create_comparison_spider_chart(
                        successful_pokemon_data_for_graph)
                    if base64_image:
                        print(f"--- {graph_label} Generated (Base64) ---")
                    else:
                        print(
                            f"--- Failed to generate chart (function returned None) ---")
                except Exception as e:
                    print(f"ERROR: Failed to generate spider chart: {e}")
                    traceback.print_exc()  # Ispis detalja iznimke
            else:
                # Slučaj kada nema uspješnih rezultata za grafikon
                print(
                    "--- No successful Pokémon data with stats found for graphing. ---")

            # 6. Dodavanje sirovih odgovora alata u listu poruka za drugi LLM poziv
            messages.extend(tool_call_responses)

            # 7. Drugi poziv OpenAI API-ju
            # LLM sada ima rezultate alata i treba generirati konačni tekstualni odgovor.
            print("--- Sending tool results back to LLM for final response ---")
            second_response = openai.chat.completions.create(
                model=GPT_MODEL,
                messages=messages  # Prosljeđivanje ažurirane liste poruka
            )
            # Dohvaćanje konačnog tekstualnog odgovora
            final_response_text = second_response.choices[0].message.content
            print(f"LLM (after tool): {final_response_text}")

            # 8. Ugrađivanje Base64 slike grafikona u Markdown formatu ako je generirana
            if base64_image:
                # Kreiranje Markdown stringa za sliku
                image_markdown = f"\n\n![{graph_label}](data:image/png;base64,{base64_image})"
                # Dodavanje Markdown stringa na kraj tekstualnog odgovora
                final_response_text += image_markdown

            # Vraćanje konačnog odgovora (tekst + slika) Gradio sučelju
            return final_response_text

        # 9. Slučaj kada LLM nije zatražio alat
        else:
            # Direktno vraćanje tekstualnog odgovora LLM-a
            final_response_text = response_message.content
            print(f"LLM (no tool): {final_response_text}")
            return final_response_text

    # Općenito rukovanje iznimkama tijekom cijelog procesa
    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        traceback.print_exc()  # Ispis stack trace-a za debugiranje
        # Vraćanje poruke o grešci korisniku
        return f"Sorry, an error occurred: {str(e)}"


# --- Pokretanje Gradio Korisničkog Sučelja ---
# Konfiguracija Gradio ChatInterface komponente
view = gr.ChatInterface(
    fn=chat,  # Povezivanje s glavnom chat funkcijom
    title="PokéPal AI - Comparison Expert",  # Naslov aplikacije
    description="Ask about a Pokémon or compare stats! E.g., 'Tell me about Pikachu' or 'Compare Pikachu and Charizard stats'.",  # Opis
    examples=["Show me the stats of Gen 1 starter Pokémons and recommend who to pick", "Pikachu vs Raichu",
              "What are Mewtwo's stats?", "Tell me about Eevee"],  # Primjeri upita
    cache_examples=False  # Onemogućavanje keširanja primjera za dinamičke odgovore
)

# Glavni blok za pokretanje aplikacije kada se skripta direktno izvršava
if __name__ == "__main__":
    view.launch()  # Pokretanje Gradio web servera

# --- END OF FILE ---
