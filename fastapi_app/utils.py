import inspect
from typing import Type

def make_fastapi_class_based_view(fastapi_app, cls: Type, ingress_instance) -> None:
    """Transform the `cls`'s methods and class annotations to FastAPI routes.

    Modified from
    https://github.com/ray-project/ray/blob/6c7da025c38d3775092d5491988289f9e63cdaff/python/ray/serve/_private/http_util.py#L296
    """
    # Delayed import to prevent ciruclar imports in workers.
    from fastapi import APIRouter, Depends
    from fastapi.routing import APIRoute, APIWebSocketRoute

    def get_current_servable_instance():
        return ingress_instance

    # Find all the class method routes
    class_method_routes = [
        route
        for route in fastapi_app.routes
        if
        # User defined routes must all be APIRoute or APIWebSocketRoute.
        isinstance(route, (APIRoute, APIWebSocketRoute))
        # We want to find the route that's bound to the `cls`.
        # NOTE(simon): we can't use `route.endpoint in inspect.getmembers(cls)`
        # because the FastAPI supports different routes for the methods with
        # same name. See #17559.
        and (cls.__qualname__ in route.endpoint.__qualname__)
    ]

    # Modify these routes and mount it to a new APIRouter.
    # We need to to this (instead of modifying in place) because we want to use
    # the laster fastapi_app.include_router to re-run the dependency analysis
    # for each routes.
    new_router = APIRouter()
    for route in class_method_routes:
        fastapi_app.routes.remove(route)

        # This block just adds a default values to the self parameters so that
        # FastAPI knows to inject the object when calling the route.
        # Before: def method(self, i): ...
        # After: def method(self=Depends(...), *, i):...
        old_endpoint = route.endpoint
        old_signature = inspect.signature(old_endpoint)
        old_parameters = list(old_signature.parameters.values())
        if len(old_parameters) == 0:
            # TODO(simon): make it more flexible to support no arguments.
            raise ValueError(
                "Methods in FastAPI class-based view must have ``self`` as "
                "their first argument."
            )
        old_self_parameter = old_parameters[0]
        new_self_parameter = old_self_parameter.replace(
            default=Depends(get_current_servable_instance)
        )
        new_parameters = [new_self_parameter] + [
            # Make the rest of the parameters keyword only because
            # the first argument is no longer positional.
            parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY)
            for parameter in old_parameters[1:]
        ]
        new_signature = old_signature.replace(parameters=new_parameters)
        setattr(route.endpoint, "__signature__", new_signature)
        setattr(route.endpoint, "_serve_cls", cls)
        new_router.routes.append(route)
    fastapi_app.include_router(new_router)

    routes_to_remove = list()
    for route in fastapi_app.routes:
        if not isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue

        # If there is a response model, FastAPI creates a copy of the fields.
        # But FastAPI creates the field incorrectly by missing the outer_type_.
        if (
            # TODO(edoakes): I don't think this check is complete because we need
            # to support v1 models in v2 (from pydantic.v1 import *).
            isinstance(route, APIRoute)
            and route.response_model
        ):
            route.secure_cloned_response_field.outer_type_ = (
                route.response_field.outer_type_
            )

        # Remove endpoints that belong to other class based views.
        serve_cls = getattr(route.endpoint, "_serve_cls", None)
        if serve_cls is not None and serve_cls != cls:
            routes_to_remove.append(route)
    fastapi_app.routes[:] = [r for r in fastapi_app.routes if r not in routes_to_remove]
