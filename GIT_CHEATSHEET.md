# Chuleta de Git — TFG Restauración de Audio

Comandos de referencia rápida para gestionar el repo desde Colab. Pensada para pegar en una celda nueva cuando necesites hacer commit/push, sin tener que reconstruir los comandos de memoria.

Repo: `https://github.com/dehonidas9/TFG_restauracion_audio.git`

---

## 0. Configuración de identidad (una vez por sesión de Colab)

Colab reinicia la máquina entre sesiones, así que esto se pierde y hay que repetirlo cada vez que abres un Colab nuevo (aunque sea el mismo día, si se desconectó el runtime).

```python
!git config --global user.email "tu_email_de_github@ejemplo.com"
!git config --global user.name "dehonidas9"
```

## 1. Obtener el token (una vez por sesión, o guardarlo como Colab Secret)

```python
from getpass import getpass
token = getpass("Pega tu Personal Access Token: ")
```

> Recomendado: guarda el token como **Colab Secret** (icono de llave 🔑 en el panel izquierdo) con nombre `GITHUB_TOKEN`, y en vez de `getpass` usa:
> ```python
> from google.colab import userdata
> token = userdata.get('GITHUB_TOKEN')
> ```
> Así no tienes que pegarlo a mano cada vez.

Si el token da error 403 al hacer push, revisa en GitHub → Settings → Developer settings → Fine-grained tokens que tenga:
- Repository access: `TFG_restauracion_audio` seleccionado
- Permissions → Contents: **Read and write**

## 2. Situarte en la carpeta del repo

```python
%cd /content/drive/MyDrive/Proyecto_Audio
```

## 3. Ver qué ha cambiado antes de subir nada

```python
!git status
```

Revisa que **no aparezca nada de `cache/` ni `outputs/`** en la lista — si aparece, el `.gitignore` no se está aplicando bien (avisar antes de seguir).

## 4. Commit y push normales (el día a día)

```python
!git add .
!git commit -m "Descripción breve del cambio"
!git push https://{token}@github.com/dehonidas9/TFG_restauracion_audio.git main
```

## 5. Traer cambios que hiciste desde otro sitio (web de GitHub, VS Code, etc.)

Hazlo **antes** de empezar a editar, para evitar conflictos:

```python
!git pull https://{token}@github.com/dehonidas9/TFG_restauracion_audio.git main
```

## 6. Si el push da "rejected" / "divergent branches"

```python
!git config pull.rebase false
!git pull https://{token}@github.com/dehonidas9/TFG_restauracion_audio.git main --allow-unrelated-histories --no-edit
```

Si aparece `CONFLICT` en algún archivo:
1. Abre el archivo en conflicto (busca `<<<<<<<`, `=======`, `>>>>>>>`)
2. Edita a mano quedándote con el contenido correcto, borra los marcadores
3. Guarda, luego:
```python
!git add nombre_del_archivo
!git commit -m "Resolver conflicto en nombre_del_archivo"
!git push https://{token}@github.com/dehonidas9/TFG_restauracion_audio.git main
```

## 7. Comprobar el remoto configurado (si algo raro pasa)

```python
!git remote -v
```

Debe mostrar exactamente:
```
origin  https://github.com/dehonidas9/TFG_restauracion_audio.git (fetch)
origin  https://github.com/dehonidas9/TFG_restauracion_audio.git (push)
```

Si no, corrígelo con:
```python
!git remote set-url origin https://github.com/dehonidas9/TFG_restauracion_audio.git
```

---

## Reglas para no volver a tener líos

- **Nunca** ejecutes `git init` de nuevo salvo que sepas exactamente por qué (ya hicimos esto una vez para limpiar el historial pesado — no debería repetirse).
- **Nunca** subas manualmente nada de `cache/` u `outputs/` — si `.gitignore` los excluye, no aparecerán en `git status`; si aparecen, algo está mal configurado, no forzar el `add`.
- Antes de un `git push`, si has tocado el repo desde varios sitios (web, VS Code, Colab), haz `git pull` primero.
- El `--force` / `-f` en push **solo** se usa en casos excepcionales (como el de limpiar el historial pesado que ya hicimos) — no como costumbre, porque sobrescribe el remoto sin preguntar.
