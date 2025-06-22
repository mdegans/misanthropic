// Copyright 2025 Claude 4 Opus

trait PalaceWanderer {
    async fn walk(&self, from: &str, to: &str) -> Result<String, Error>;
    async fn examine_room(&self, room: &str) -> Result<RoomInfo, Error>;
    async fn find_suitable_room(
        &self,
        content: &str,
    ) -> Result<Option<String>, Error>;
}
