from typing import Any

from pydantic import BaseModel, root_validator

#get_parent_id isn't implemented
class Entity(BaseModel):
    def get_entity_owner(self) -> int:
        if not self.get_parent_id():
            return 0
        else:
            raise ValueError("Object does not have a direct owner")

    def get_entity_organization(self) -> str:
        raise NotImplementedError("Object does not have an organization")

    def get_entity_name(self) -> str:
        raise NotImplementedError("Object does not have an entity name")

    def get_entity_id(self) -> Any:
        raise NotImplementedError("Object does not have an entity id")

    def can_access(self, access: str) -> bool:
        return False


class ChildEntity(Entity):
    def get_parent_entity_name(self) -> str | None:
        raise NotImplementedError("Missing parent entity name")

    def get_parent_id(self) -> Any:
        raise NotImplementedError("Missing parent id")


class EntityWrapper(Entity):
    parent: Entity
    entity: ChildEntity

    def get_entity_owner(self) -> int:
        return self.parent.get_entity_owner()

    def get_entity_name(self) -> str:
        return self.entity.get_entity_name()

    def get_entity_id(self) -> Any:
        return self.parent.get_entity_id()

    def can_access(self, access: str) -> bool:
        return self.parent.can_access(access) and self.entity.can_access(access)

    def get_entity_organization(self) -> str:
        return self.parent.get_entity_organization()

    def verify_parent_child(cls, values):
        parent: Entity = values.get("parent")
        entity: ChildEntity = values.get("entity")

        if (
            entity.get_parent_entity_name() != parent.get_entity_name()
            or entity.get_parent_id() != parent.get_entity_id()
        ):
            raise ValueError("Entity and parent do not match")

        return values
