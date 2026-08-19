# The Angular dev server.
#
# Deliberately `ng serve` rather than a production nginx build: this stack exists
# to be clicked and changed, and a static bundle would mean rebuilding an image
# for every edit. The API client is generated (R7.3), so `npx orval` can be run
# in here after the backend schema changes.

FROM node:24-alpine

WORKDIR /app

# Dependencies first: editing a component should not reinstall Angular.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ENV LLMDEX_API_TARGET=http://backend:8001

EXPOSE 4200

# --poll because file-change events do not cross a bind mount on Windows or
# macOS, so without it edits on the host never reach the watcher.
CMD ["npx", "ng", "serve", "--host", "0.0.0.0", "--port", "4200", "--poll", "2000"]
