from src.agent import create_agent
from src.prompts import FALLBACK_RESPONSE


def main():
    print("=" * 62)
    print("  URBANSTYLE - Agente de Atención Inteligente (Eva 2)")
    print("  Herramientas: consulta | razonamiento | escritura")
    print("  Memoria de corto y largo plazo activada")
    print("  Escribe 'salir' para terminar | 'memoria' para ver hechos")
    print("=" * 62)
    print("\nInicializando agente...")
    agent = create_agent()
    print(f"Sesiones previas registradas: {agent.long_mem.data.get('sessions', 1)}")
    print("Listo. Haz tu consulta.\n")

    while True:
        user_input = input("Tú: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            print("\nGracias por contactar UrbanStyle. ¡Hasta pronto!")
            break
        if user_input.lower() == "memoria":
            print("\n[MEMORIA LARGO PLAZO]")
            print(agent.long_mem.render_facts())
            print(f"\nSesiones: {agent.long_mem.data.get('sessions', 0)}")
            continue

        try:
            result = agent.run(user_input)
            print(f"\nUrbanAgent: {result['answer']}")
            print(f"  [plan: {result['plan'].intent} | herramientas: {', '.join(result['tools_used']) or 'ninguna'}]")
        except Exception as e:
            print(f"\nUrbanAgent: {FALLBACK_RESPONSE}")
            print(f"  [Error técnico: {e}]")
        print()


if __name__ == "__main__":
    main()
